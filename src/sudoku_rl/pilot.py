"""Lightweight pilot runner for short Sudoku RL experiments.

Use this to compare small configuration changes (e.g., terminate_on_wrong_digit)
without touching the main training script. Keeps defaults small so runs are cheap.

Supports running multiple bins and multiple variants in one invocation; each
variant/bin combo gets its own TensorBoard subdirectory for easy comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
import logging
import math

import numpy as np
import torch
import pufferlib.vector
from pufferlib import pufferl

from .make_vecenv import make_sudoku_vecenv
from .sudoku_mlp import SudokuMLP


# --- Minor helpers (duplicated from train.py to stay self-contained) ---


def _parse_bin(label: str) -> tuple[int, int]:
    parts = label.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected bin label format: {label}")
    return int(parts[1]), int(parts[2])


def max_steps_for_bin(label: str, fudge: float = 1.2) -> int:
    lo, hi = _parse_bin(label)
    empties = hi
    return int(np.ceil(empties * 9 * fudge))


class TensorboardLogger:
    """Minimal logger matching PuffeRL's expected interface."""

    def __init__(self, logdir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(logdir)
        self.run_id = str(int(time.time()))

    def log(self, logs, step: int):
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def close(self, model_path=None):
        self.writer.flush()
        self.writer.close()


def load_base_config(device: str, total_steps: int, num_envs: int, bptt_horizon: int, minibatch_size: int):
    """Load PuffeRL base config with Sudoku-friendly defaults."""

    # PuffeRL inspects sys.argv inside load_config; isolate it.
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        cfg = pufferl.load_config("puffer_breakout")
    finally:
        sys.argv = original_argv

    cfg["train"]["device"] = device
    cfg["train"]["total_timesteps"] = total_steps
    cfg["vec"]["num_envs"] = num_envs
    cfg["train"]["bptt_horizon"] = bptt_horizon

    batch_size = num_envs * bptt_horizon
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["minibatch_size"] = min(minibatch_size, batch_size)
    cfg["train"]["max_minibatch_size"] = cfg["train"]["minibatch_size"]

    cfg["rnn_name"] = None
    cfg["train"]["use_rnn"] = False
    cfg["train"]["env"] = "sudoku"
    return cfg


def patch_sps_property():
    """Avoid transient zero SPS reports when logging very frequently."""

    def _patched_sps(self):
        raw = 0
        if self.global_step != self.last_log_step:
            raw = (self.global_step - self.last_log_step) / max(1e-6, (time.time() - self.last_log_time))
        if raw == 0 and hasattr(self, "_prev_sps") and self._prev_sps:
            return self._prev_sps
        self._prev_sps = raw
        return raw

    pufferl.PuffeRL.sps = property(_patched_sps)


def run_single(cfg, args, bin_label: str, terminate_on_wrong_digit: bool, run_id: str):
    # Keep noisy library warnings down for quick pilots
    warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available.*", category=UserWarning)
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

    backend_map = {
        "serial": pufferlib.vector.Serial,
        "mp": pufferlib.vector.Multiprocessing,
    }
    backend_cls = backend_map[args.backend]

    vec_batch_size = min(args.vec_batch_size, args.num_envs)
    if args.backend == "mp" and args.num_envs % vec_batch_size != 0:
        vec_batch_size = math.gcd(args.num_envs, vec_batch_size) or args.num_envs
    vec_zero_copy = args.vec_zero_copy and args.backend == "mp" and args.num_envs % vec_batch_size == 0
    vec_overwork = args.vec_overwork if args.backend == "mp" else False

    steps_budget = args.total_steps
    max_steps = args.max_steps or max_steps_for_bin(bin_label, args.max_steps_fudge)

    vecenv = make_sudoku_vecenv(
        bin_label,
        num_envs=args.num_envs,
        seed=args.seed,
        max_steps=max_steps,
        backend=backend_cls,
        num_workers=args.num_workers,
        vec_batch_size=vec_batch_size,
        vec_zero_copy=vec_zero_copy,
        vec_overwork=vec_overwork,
        terminate_on_wrong_digit=terminate_on_wrong_digit,
    )

    policy = SudokuMLP(vecenv.driver_env).to(cfg["train"]["device"])

    logdir = None
    if args.tb_logdir:
        logdir = f"{args.tb_logdir}/{run_id}"
    tb_logger = TensorboardLogger(logdir) if logdir else None

    patch_sps_property()
    algo = pufferl.PuffeRL(cfg["train"], vecenv, policy, logger=tb_logger)

    log_step = args.log_every
    next_log = algo.global_step + log_step

    print(
        f"Pilot run — bin={bin_label}, terminate_on_wrong_digit={terminate_on_wrong_digit}, "
        f"steps={steps_budget}, envs={args.num_envs}, logdir={logdir}"
    )

    while algo.global_step < steps_budget:
        algo.evaluate()
        algo.train()

        if algo.global_step >= next_log:
            algo.print_dashboard()
            next_log += log_step

    # Final snapshot
    algo.print_dashboard()
    if tb_logger:
        tb_logger.close()
    vecenv.close()

    print("Done. Final stats:", {k: float(v) if hasattr(v, "__float__") else v for k, v in algo.last_stats.items()})


def main():
    parser = argparse.ArgumentParser(description="Lightweight Sudoku RL pilot runner")
    parser.add_argument("--bins", nargs="+", default=["zeros_08_11"], help="puzzle bin labels (space-separated)")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--bptt_horizon", type=int, default=32)
    parser.add_argument("--minibatch_size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", choices=["serial", "mp"], default="serial")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--vec_batch_size", type=int, default=64, help="Vecenv batch size for MP backend")
    parser.add_argument("--vec_zero_copy", action="store_true", default=False)
    parser.add_argument("--vec_overwork", action="store_true", default=False)
    parser.add_argument("--log_every", type=int, default=2_000)
    parser.add_argument("--tb_logdir", default="runs/pilot", help="TensorBoard log dir (empty to disable)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None, help="Override per-episode step limit")
    parser.add_argument("--max_steps_fudge", type=float, default=1.2, help="Multiplier for heuristic max_steps")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["terminate", "no-terminate"],
        choices=["terminate", "no-terminate"],
        help="Which variants to run (terminate = end on code 7)"
    )

    args = parser.parse_args()
    cfg = load_base_config(
        device=args.device,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        bptt_horizon=args.bptt_horizon,
        minibatch_size=args.minibatch_size,
    )

    patch_sps_property()

    runs = []
    for bin_label in args.bins:
        for variant in args.variants:
            term_flag = variant == "terminate"
            run_id = f"{variant}/{bin_label}"
            runs.append((bin_label, term_flag, run_id))

    for idx, (bin_label, term_flag, run_id) in enumerate(runs):
        print(f"\n=== Run {idx+1}/{len(runs)}: {run_id} ===")
        run_single(cfg, args, bin_label, term_flag, run_id)


if __name__ == "__main__":
    main()
