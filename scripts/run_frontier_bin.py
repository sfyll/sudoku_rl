#!/usr/bin/env python
"""
Run a focused single-bin PPO training with distance-based shaping.
Defaults mirror scripts/run_cpu_defaults.sh (CPU, serial backend) but lock to one bin
and no curriculum, to study frontier-bin behaviour and ΔF signal quality.
"""
from __future__ import annotations
import argparse
import sys
import time
import math
import logging
import warnings
import os
from pathlib import Path

import numpy as np
import torch
import psutil
import pufferlib.vector
from pufferlib import pufferl

from sudoku_rl.make_vecenv import make_sudoku_vecenv
from sudoku_rl.sudoku_mlp import SudokuMLP


def patch_sps_property():
    def _patched_sps(self):
        raw = 0
        if self.global_step != self.last_log_step:
            raw = (self.global_step - self.last_log_step) / max(1e-6, (time.time() - self.last_log_time))
        if raw == 0 and hasattr(self, "_prev_sps") and self._prev_sps:
            return self._prev_sps
        self._prev_sps = raw
        return raw
    pufferl.PuffeRL.sps = property(_patched_sps)


def load_base_config(device: str, total_steps: int, num_envs: int, bptt_horizon: int, minibatch_size: int):
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
    # match train.py hyperparams
    cfg["train"].update(
        learning_rate=3e-3,
        gae_lambda=0.95,
        update_epochs=4,
        ent_coef=0.0003,
        gamma=0.995,
        vf_coef=0.8,
        clip_coef=0.2,
        compile=False,
        compile_fullgraph=False,
    )
    if cfg["train"]["total_timesteps"] < cfg["train"]["batch_size"]:
        cfg["train"]["total_timesteps"] = cfg["train"]["batch_size"]
    return cfg


def max_steps_for_bin(label: str, fudge: float = 1.2) -> int:
    parts = label.split("_")
    hi = int(parts[-1]) if parts and parts[-1].isdigit() else 40
    return int(np.ceil(hi * 9 * fudge))


def main():
    parser = argparse.ArgumentParser(description="Single-bin frontier run (distance shaping)")
    parser.add_argument("--bin", dest="bin_label", default="zeros_32_35", help="Target bin (no curriculum)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["serial", "mp"], default="serial")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--bptt_horizon", type=int, default=32)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--log_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tb_logdir", default=None, help="TensorBoard dir (set to enable; default runs/frontier/<timestamp>)")
    parser.add_argument("--max_steps_fudge", type=float, default=1.2)
    parser.add_argument("--distance_model", type=Path, default=Path("experiments/distance_regressor.pt"))
    parser.add_argument("--calibrator", type=Path, default=Path("experiments/distance_calibrator.json"))
    args = parser.parse_args()

    # warnings down
    warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available.*", category=UserWarning)
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    # Safe cpu_count (macOS permission issues)
    _psutil_cpu_count = psutil.cpu_count
    def _safe_cpu_count(*a, **k):
        try:
            return _psutil_cpu_count(*a, **k)
        except Exception:
            return os.cpu_count() or 1
    psutil.cpu_count = _safe_cpu_count

    cfg = load_base_config(args.device, args.total_steps, args.num_envs, args.bptt_horizon, args.minibatch_size)

    backend_map = {
        "serial": pufferlib.vector.Serial,
        "mp": pufferlib.vector.Multiprocessing,
    }
    backend_cls = backend_map[args.backend]

    max_steps = max_steps_for_bin(args.bin_label, args.max_steps_fudge)
    vec_batch_size = min(args.num_envs, 256)  # keep modest for MP; ignored for serial
    vec_zero_copy = False
    vec_overwork = False

    vecenv = make_sudoku_vecenv(
        args.bin_label,
        num_envs=args.num_envs,
        seed=args.seed,
        max_steps=max_steps,
        backend=backend_cls,
        num_workers=2 if args.backend == "mp" else None,
        vec_batch_size=vec_batch_size if args.backend == "mp" else None,
        vec_zero_copy=vec_zero_copy,
        vec_overwork=vec_overwork,
        prev_mix_ratio=0.0,
    )

    # inject artifact paths into driver env
    driver_env = vecenv.driver_env
    driver_env.env.distance_model = driver_env.env._load_distance_model(args.distance_model, driver_env.env.device)
    driver_env.env.calibrator = driver_env.env._load_calibrator(args.calibrator)
    driver_env.env.current_F = driver_env.env._predict_F(driver_env.env.board)
    driver_env.env.start_F = driver_env.env.current_F

    policy = SudokuMLP(vecenv.driver_env).to(cfg["train"]["device"])

    patch_sps_property()
    run_dir = Path(args.tb_logdir) if args.tb_logdir else Path("runs/frontier") / f"{args.bin_label}_{int(time.time())}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = type("TBWrap", (), {
        "writer": SummaryWriter(run_dir),
        "run_id": str(int(time.time())),
        "log": lambda self, logs, step: [self.writer.add_scalar(k, v, step) for k, v in logs.items() if isinstance(v, (int, float))],
        "close": lambda self, model_path=None: (self.writer.flush(), self.writer.close()),
    })()

    algo = pufferl.PuffeRL(cfg["train"], vecenv, policy, logger=tb_logger)

    next_log = algo.global_step + args.log_every
    print(f"Frontier run — bin={args.bin_label}, device={args.device}, envs={args.num_envs}, steps={args.total_steps}")

    while algo.global_step < args.total_steps:
        algo.evaluate()
        algo.train()
        if algo.global_step >= next_log:
            algo.print_dashboard()
            next_log += args.log_every

    algo.print_dashboard()
    if tb_logger:
        tb_logger.close()
    vecenv.close()


if __name__ == "__main__":
    main()
