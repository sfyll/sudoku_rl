# src/sudoku_rl/train_with_pufferl.py
import argparse
import sys

import pufferlib.vector

from pufferlib import pufferl  # their trainer module
# from pufferlib import models  # their default policy module

from .make_vecenv import make_sudoku_vecenv
from .sudoku_mlp import SudokuMLP
from .env_puffer import SudokuPufferEnv

import imageio

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
    args = parser.parse_args()

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

    max_env_steps = args.bptt_horizon

    backend_map = {
        "serial": pufferlib.vector.Serial,
        "mp": pufferlib.vector.Multiprocessing,
    }
    backend_cls = backend_map[args.backend]

    # 2) Make vecenv for the first phase (easy puzzles)
    vecenv = make_sudoku_vecenv(
        "super-easy",
        num_envs=args.num_envs,
        max_steps=max_env_steps,
        backend=backend_cls,
        num_workers=args.num_workers,
    )

    # Keep our Sudoku-specific policy instead of replacing it with the
    # default Breakout policy referenced by the PuffeRL config. We still
    # reuse the rest of the hyperparameters (batch sizes, optimizer, etc.).
    policy = SudokuMLP(vecenv.driver_env)
    policy = policy.to(cfg["train"]["device"])

    # 4) Create PuffeRL trainer
    algo = pufferl.PuffeRL(cfg["train"], vecenv, policy)

    # === Easy phase ===
    print(f"{algo.global_step=}")
    print(f"{args.easy_steps=}")
    log_step = args.log_every
    solved_window = []
    while algo.global_step < args.super_easy_steps:
        algo.evaluate()     # collect rollouts
        algo.train()        # do updates

        if algo.global_step % log_step == 0:
            solved_mean = algo.stats.get("solved_episode", 0)
            solved_window.append(solved_mean)
            if len(solved_window) > args.early_stop_window:
                solved_window.pop(0)
            if (
                len(solved_window) == args.early_stop_window
                and all(v >= args.early_stop_threshold for v in solved_window)
            ):
                print(
                    f"Early stop: solved_episode window >= {args.early_stop_threshold} "
                    f"for {args.early_stop_window} logs"
                )
                break
            algo.print_dashboard()

    # Optional eval recording once we've finished/early-stopped super-easy
    if args.record_frames:
        record_policy_run(
            policy=policy,
            device=cfg["train"]["device"],
            difficulty="super-easy",
            max_steps=max_env_steps,
            frame_count=args.record_frames_count,
            gif_path=args.record_gif_path,
            fps=args.record_fps,
        )

    print(f"over")
    
    # === Easy phase ===
    vecenv.close()
    #vecenv = make_sudoku_vecenv(
    #    "easy",
    #    num_envs=args.num_envs,
    #    max_steps=max_env_steps,
    #)
    #algo.vecenv = vecenv  # depending on their API you may have to rebuild algo

    #while algo.global_step < args.easy_steps:
    #    algo.evaluate()     # collect rollouts
    #    algo.train()        # do updates
    #    algo.print_dashboard()

    ## === Medium phase ===
    #vecenv.close()
    #vecenv = make_sudoku_vecenv(
    #    "medium",
    #    num_envs=args.num_envs,
    #    max_steps=max_env_steps,
    #)
    #algo.vecenv = vecenv  # depending on their API you may have to rebuild algo

    #while algo.global_step < args.medium_steps:
    #    algo.evaluate()
    #    algo.train()
    #    algo.print_dashboard()

    ## === Hard phase ===
    #vecenv.close()
    #vecenv = make_sudoku_vecenv(
    #    "hard",
    #    num_envs=args.num_envs,
    #    max_steps=max_env_steps,
    #)
    #algo.vecenv = vecenv

    #while algo.global_step < args.total_steps:
    #    algo.evaluate()
    #    algo.train()
    #    algo.print_dashboard()

    #algo.save_checkpoint()
    #vecenv.close()

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
