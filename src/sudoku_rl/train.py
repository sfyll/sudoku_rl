# src/sudoku_rl/train_with_pufferl.py
import warnings
import logging
# Mute noisy warnings before importing torch
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Redirects are currently not supported in Windows or MacOs.*", category=UserWarning)
# Silence torch elastic redirect logger noise
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import psutil

# Monkeypatch psutil.cpu_count to avoid macOS permission errors inside PuffeRL's Utilization thread
_psutil_cpu_count = psutil.cpu_count
def _safe_cpu_count(*args, **kwargs):
    try:
        return _psutil_cpu_count(*args, **kwargs)
    except Exception:
        return os.cpu_count() or 1
psutil.cpu_count = _safe_cpu_count

import argparse
import sys
import os
import math
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

import pufferlib.vector

from pufferlib import pufferl  # their trainer module
# from pufferlib import models  # their default policy module

from .make_vecenv import make_sudoku_vecenv
from .sudoku_mlp import SudokuMLP
from .env_puffer import SudokuPufferEnv
from .puzzle import supported_bins, sample_puzzle, preload_puzzle_cache
from .curriculum import build_default_buckets
from .return_stats import SharedReturnStatsRegistry
from pathlib import Path

import imageio


def _warm_start_resources(distance_state):
    """Preload puzzle cache and distance model once in the parent process."""
    preload_puzzle_cache()
    # Touch distance model/calibrator once; failures are non-fatal.
    try:
        board, solution = sample_puzzle(return_solution=True, prev_mix_ratio=0.0)
        from .env import SudokuEnv
        env = SudokuEnv(
            initial_board=board,
            solution_board=solution,
            max_steps=1,
            distance_device="cpu",
            distance_state=distance_state,
        )
        env.reset()
    except Exception:
        # Warm start is best-effort; ignore errors so training can proceed.
        pass
def _parse_bin(label: str) -> tuple[int, int]:
    parts = label.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected bin label format: {label}")
    return int(parts[1]), int(parts[2])


def max_steps_for_bin(label: str, fudge: float = 1.2) -> int:
    lo, hi = _parse_bin(label)
    empties = hi
    # Rough heuristic: ~9 actions per empty cell with a small buffer
    return int(np.ceil(empties * 9 * fudge))


class TensorboardLogger:
    """Minimal logger that matches PuffeRL's expected interface."""

    def __init__(self, logdir: str):
        self.writer = SummaryWriter(logdir)
        self.run_id = str(int(time.time()))

    def log(self, logs, step: int):
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def close(self, model_path=None):
        self.writer.flush()
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device string (cpu, cuda, cuda:1, mps, or auto)")
    parser.add_argument("--total_steps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--bptt_horizon", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--backend", type=str, default="mp", choices=["serial", "mp"], help="Vecenv backend (curriculum currently expects serial)")
    parser.add_argument("--num_workers", type=int, default=16, help="Workers for mp backend (set lower if you saturate cores)")
    parser.add_argument("--vec_batch_size", type=int, default=256, help="Vecenv batch size for MP backend (controls worker barrier)")
    parser.add_argument("--vec_zero_copy", action="store_true", default=False, help="Use zero_copy (contiguous workers, no memcpy). Default False to allow non-contiguous copies")
    parser.add_argument("--vec_overwork", action="store_true", default=False, help="Allow num_workers > physical cores (PufferLib overwork)")
    parser.add_argument("--log_every", type=int, default=5000, help="Print dashboard every N global steps")
    parser.add_argument("--record_frames", action="store_true", help="Enable PuffeRL frame recording/gif output")
    parser.add_argument("--record_frames_count", type=int, default=200, help="How many frames to capture when recording")
    parser.add_argument("--record_gif_path", type=str, default="experiments/sudoku_eval.gif", help="Where to write the gif when recording")
    parser.add_argument("--record_fps", type=int, default=10, help="FPS for the recorded gif")
    parser.add_argument("--tb_logdir", type=str, default="runs/sudoku", help="TensorBoard log directory (set empty to disable)")
    args = parser.parse_args()

    # Mute noisy warnings (pynvml deprecation, cuda-not-available on CPU/MPS, torch elastic redirects)
    warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="Redirects are currently not supported in Windows or MacOs.*", category=UserWarning)

    # 1) Load a base config from some existing env (e.g. breakout) and tweak
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        cfg = pufferl.load_config("puffer_breakout")  # good baseline hyperparams
    finally:
        sys.argv = original_argv
    if args.device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = args.device

    cfg["train"]["device"] = resolved_device          # e.g. "cpu", "cuda", "cuda:1", "mps"
    cfg["train"]["total_timesteps"] = args.total_steps
    cfg["vec"]["num_envs"] = args.num_envs
    cfg["train"]["bptt_horizon"] = args.bptt_horizon

    batch_size = args.num_envs * args.bptt_horizon
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["minibatch_size"] = min(args.minibatch_size, batch_size)
    cfg["train"]["max_minibatch_size"] = cfg["train"]["minibatch_size"]
    cfg["rnn_name"] = None
    cfg["train"]["use_rnn"] = False
    cfg["train"]["env"] = "sudoku"
    
    # PPO CONFIG
    cfg["train"]["learning_rate"] = 3e-3 # reduce to something smaller
    cfg["train"]["gae_lambda"] = 0.95 #bias-variance tradeoff. 0.95 given long-horizon problem
    cfg["train"]["update_epochs"] = 4 # update more aggressively per batch
    cfg["train"]["ent_coef"] = 0.0003 #more exploratory
    cfg["train"]["gamma"] = 0.995
    cfg["train"]["vf_coef"] = 0.8
    cfg["train"]["clip_coef"] = 0.2
    cfg["train"]["compile"] = False 
    # Allow graph breaks (e.g., numpy-based action masks); fullgraph would fail on them.
    cfg["train"]["compile_fullgraph"] = False

    # Optional recording: PuffeRL uses base-level flags for frame saving
    cfg["base"]["save_frames"] = args.record_frames

    backend_map = {
        "serial": pufferlib.vector.Serial,
        "mp": pufferlib.vector.Multiprocessing,
    }
    backend_cls = backend_map[args.backend]

    # Tune vecenv batching to reduce recv barrier in MP backend
    vec_batch_size = min(args.vec_batch_size, args.num_envs)
    if args.backend == "mp" and args.num_envs % vec_batch_size != 0:
        # Ensure divisibility; fallback to greatest common divisor to avoid zero_copy assertions
        vec_batch_size = math.gcd(args.num_envs, vec_batch_size) or args.num_envs
    vec_zero_copy = args.vec_zero_copy and (args.backend == "mp") and (args.num_envs % vec_batch_size == 0)
    vec_overwork = args.vec_overwork if args.backend == "mp" else False

    # Load distance model weights once on CPU and hand the state_dict to workers
    # to avoid each process reading from disk.
    distance_model_path = Path("experiments/distance_regressor.pt")
    distance_state = torch.load(distance_model_path, map_location="cpu")
    _warm_start_resources(distance_state)

    bins = list(supported_bins())
    if not bins:
        raise RuntimeError("No sudoku bins available. Generate data with scripts/create_filtered_dataset_sudoku.py")

    bucket_defs = build_default_buckets(bins)
    # Shared return stats for per-bucket reward scaling across env workers
    shared_return_stats = SharedReturnStatsRegistry(
        num_buckets=len(bucket_defs),
        target_std=5.0,
        scale_min=0.1,
        scale_max=10.0,
        eps=1e-6,
        min_episodes_for_scale=50,
    )
    hardest_bin = bucket_defs[-1].bin_label
    vecenv = make_sudoku_vecenv(
        bucket_defs[0].bin_label,
        num_envs=args.num_envs,
        max_steps=max_steps_for_bin(hardest_bin),
        backend=backend_cls,
        num_workers=args.num_workers,
        vec_batch_size=vec_batch_size,
        vec_zero_copy=vec_zero_copy,
        vec_overwork=vec_overwork,
        prev_mix_ratio=0.0,
        bucket_defs=bucket_defs,
        shared_return_stats=shared_return_stats,
        distance_state=distance_state,
        curriculum_kwargs=dict(
            initial_unlocked=2,
            window_size=200,
            promote_thresholds=[0.9, 0.8, 0.6],
            min_episodes_for_decision=100,
            alpha=2.0,
            eps=0.05,
        ),
    )

    # Keep our Sudoku-specific policy instead of replacing it with the
    # default Breakout policy referenced by the PuffeRL config. We still
    # reuse the rest of the hyperparameters (batch sizes, optimizer, etc.).
    policy = SudokuMLP(vecenv.driver_env)
    policy = policy.to(cfg["train"]["device"])

    # 4) Create PuffeRL trainer (inject TensorBoard logger if requested)
    tb_logger = TensorboardLogger(args.tb_logdir) if args.tb_logdir else None
    algo = pufferl.PuffeRL(cfg["train"], vecenv, policy, logger=tb_logger)
    # Patch SPS reporting to hold the last non-zero value instead of 0 when
    # logs happen too frequently (avoids misleading dashboard zeros).
    def _patched_sps(self):
        raw = 0
        if self.global_step != self.last_log_step:
            raw = (self.global_step - self.last_log_step) / max(1e-6, (time.time() - self.last_log_time))
        if raw == 0 and hasattr(self, "_prev_sps") and self._prev_sps:
            return self._prev_sps
        self._prev_sps = raw
        return raw

    pufferl.PuffeRL.sps = property(_patched_sps)

    log_step = args.log_every
    next_log = log_step
    while algo.global_step < args.total_steps:
        algo.evaluate()
        algo.train()

        while algo.global_step >= next_log:
            algo.print_dashboard()
            next_log += log_step

    if tb_logger:
        tb_logger.close()
    vecenv.close()

if __name__ == "__main__":
    main()


def record_policy_run(
    policy,
    device: str,
    bin_label: str,
    max_steps: int,
    frame_count: int,
    gif_path: str,
    fps: int,
):
    """Run a single-agent eval rollout and save a GIF using env.render("rgb_array")."""
    env = SudokuPufferEnv(bin_label=bin_label, max_steps=max_steps)
    policy.eval()

    frames = []
    obs, _ = env.reset()
    with torch.no_grad():
        while len(frames) < frame_count:
            # Render frame
            frame = env.render(mode="rgb_array")
            frames.append(frame)

            # Forward + greedy action
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits, _ = policy.forward_eval(obs_t)
            action = torch.argmax(logits, dim=-1).cpu().numpy()
            # Step
            obs, rew, done, trunc, infos = env.step(action)
            if done[0]:
                obs, _ = env.reset()

    imageio.mimsave(gif_path, frames, fps=fps)
