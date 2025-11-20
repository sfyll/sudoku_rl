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
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter

import pufferlib.vector

from pufferlib import pufferl  # their trainer module
# from pufferlib import models  # their default policy module

from .make_vecenv import make_sudoku_vecenv
from .sudoku_mlp import SudokuMLP
from .env_puffer import SudokuPufferEnv

import imageio

# Heuristic per-difficulty step caps: ~9 actions per empty cell with a small buffer.
DIFFICULTY_MAX_STEPS = {
    "super-easy": 60,   # <=4 holes  -> ~36 steps, give some slack
    "easy": 300,        # <=25 holes -> ~225 steps
    "medium": 540,      # <=45 holes -> ~405 steps
    "hard": 720,        # <=60 holes -> ~540 steps
}


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
    parser.add_argument("--device", default="mps")
    parser.add_argument("--super_easy_steps", type=int, default=200_000)
    parser.add_argument("--easy_steps", type=int, default=200_000)
    parser.add_argument("--medium_steps", type=int, default=400_000)
    parser.add_argument("--total_steps", type=int, default=600_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--bptt_horizon", type=int, default=16)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--backend", type=str, default="mp", choices=["serial", "mp"], help="Vecenv backend")
    parser.add_argument("--num_workers", type=int, default=8, help="Workers for threaded/mp backends")
    parser.add_argument("--log_every", type=int, default=500, help="Print dashboard every N global steps")
    parser.add_argument("--early_stop_window", type=int, default=5, help="Stop training when solved_episode >= threshold for this many consecutive logs")
    parser.add_argument("--early_stop_threshold", type=float, default=0.99, help="Solved_episode threshold for early stopping")
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
    cfg["train"]["device"] = args.device          # e.g. "cpu" or "mps"
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

    # Optional recording: PuffeRL uses base-level flags for frame saving
    cfg["base"]["save_frames"] = args.record_frames

    backend_map = {
        "serial": pufferlib.vector.Serial,
        "mp": pufferlib.vector.Multiprocessing,
    }
    backend_cls = backend_map[args.backend]

    # 2) Make vecenv for the first phase (super-easy puzzles)
    vecenv = make_sudoku_vecenv(
        "super-easy",
        num_envs=args.num_envs,
        max_steps=DIFFICULTY_MAX_STEPS["super-easy"],
        backend=backend_cls,
        num_workers=args.num_workers,
    )

    # Keep our Sudoku-specific policy instead of replacing it with the
    # default Breakout policy referenced by the PuffeRL config. We still
    # reuse the rest of the hyperparameters (batch sizes, optimizer, etc.).
    policy = SudokuMLP(vecenv.driver_env)
    policy = policy.to(cfg["train"]["device"])

    # 4) Create PuffeRL trainer (inject TensorBoard logger if requested)
    tb_logger = TensorboardLogger(args.tb_logdir) if args.tb_logdir else None
    algo = pufferl.PuffeRL(cfg["train"], vecenv, policy, logger=tb_logger)

    # === Curriculum phases ===
    print(f"{algo.global_step=}")
    print(f"{args.easy_steps=}")
    log_step = args.log_every

    def run_phase(difficulty: str, phase_steps: int, max_steps: int):
        nonlocal vecenv
        _ = max_steps  # kept for clarity; env already set with this budget
        phase_end = algo.global_step + phase_steps
        next_log = algo.global_step + log_step
        solved_window = []

        while algo.global_step < phase_end:
            # Collect rollouts
            algo.evaluate()

            # Capture solve ratio before train clears stats
            solved_mean = 0.0
            if "solved_episode" in algo.stats and len(algo.stats["solved_episode"]):
                solved_mean = float(np.mean(algo.stats["solved_episode"]))

            # PPO updates
            logs = algo.train()

            # Prefer aggregated metrics
            if logs and "environment/solved_episode" in logs:
                solved_mean = float(logs["environment/solved_episode"])
            elif algo.last_stats and "solved_episode" in algo.last_stats:
                solved_mean = float(algo.last_stats["solved_episode"])

            while algo.global_step >= next_log:
                solved_window.append(solved_mean)
                if len(solved_window) > args.early_stop_window:
                    solved_window.pop(0)
                if (
                    len(solved_window) == args.early_stop_window
                    and all(v >= args.early_stop_threshold for v in solved_window)
                ):
                    print(
                        f"[{difficulty}] Early stop: solved_episode window >= {args.early_stop_threshold} "
                        f"for {args.early_stop_window} logs"
                    )
                    return
                algo.print_dashboard()
                next_log += log_step

    # Super-easy phase
    run_phase("super-easy", args.super_easy_steps, DIFFICULTY_MAX_STEPS["super-easy"])

    # Easy phase: switch env with larger step budget
    vecenv.close()
    vecenv = make_sudoku_vecenv(
        "easy",
        num_envs=args.num_envs,
        max_steps=DIFFICULTY_MAX_STEPS["easy"],
        backend=backend_cls,
        num_workers=args.num_workers,
    )
    vecenv.async_reset(seed=0)
    algo.vecenv = vecenv
    run_phase("easy", args.easy_steps, DIFFICULTY_MAX_STEPS["easy"])

    # Optional eval recording once we've finished/early-stopped super-easy
    if args.record_frames:
        record_policy_run(
            policy=policy,
            device=cfg["train"]["device"],
            difficulty="super-easy",
            max_steps=DIFFICULTY_MAX_STEPS["super-easy"],
            frame_count=args.record_frames_count,
            gif_path=args.record_gif_path,
            fps=args.record_fps,
        )

    print("Training finished (super-easy + easy phases)")
    if tb_logger:
        tb_logger.close()
    vecenv.close()

if __name__ == "__main__":
    main()


def record_policy_run(
    policy,
    device: str,
    difficulty: str,
    max_steps: int,
    frame_count: int,
    gif_path: str,
    fps: int,
):
    """Run a single-agent eval rollout and save a GIF using env.render("rgb_array")."""
    env = SudokuPufferEnv(difficulty=difficulty, max_steps=max_steps)
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
