# src/sudoku_rl/train_with_pufferl.py
import argparse
import sys

from pufferlib import pufferl  # their trainer module
# from pufferlib import models  # their default policy module

from .make_vecenv import make_sudoku_vecenv
from .sudoku_mlp import SudokuMLP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--super_easy_steps", type=int, default=1_000_000)
    parser.add_argument("--easy_steps", type=int, default=1_000_000)
    parser.add_argument("--medium_steps", type=int, default=2_000_000)
    parser.add_argument("--total_steps", type=int, default=3_000_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--bptt_horizon", type=int, default=32)
    parser.add_argument("--minibatch_size", type=int, default=512)
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

    max_env_steps = args.bptt_horizon

    # 2) Make vecenv for the first phase (easy puzzles)
    vecenv = make_sudoku_vecenv(
        "super-easy",
        num_envs=args.num_envs,
        max_steps=max_env_steps,
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
    while algo.global_step < args.super_easy_steps:
        algo.evaluate()     # collect rollouts
        algo.train()        # do updates
        algo.print_dashboard()
    
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
