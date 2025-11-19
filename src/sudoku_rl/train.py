# src/sudoku_rl/train_with_pufferl.py
import argparse

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
    parser.add_argument("--num_envs", type=int, default=1024)
    args = parser.parse_args()

    # 1) Load a base config from some existing env (e.g. breakout) and tweak
    cfg = pufferl.load_config("puffer_breakout")  # good baseline hyperparams
    cfg["train"]["device"] = args.device          # e.g. "mps" on your Mac
    cfg["train"]["total_timesteps"] = args.total_steps
    cfg["vec"]["num_envs"] = 2 
    cfg["train"]["env"] = "sudoku"

    # 2) Make vecenv for the first phase (easy puzzles)
    vecenv = make_sudoku_vecenv("super-easy", num_envs=args.num_envs)
    policy = SudokuMLP(vecenv.driver_env)

    # 3) Use their default policy for flat obs + discrete actions
    #    The exact call depends on models.py; something like:
    # policy = models.DefaultPolicy(cfg, vecenv)

    policy = pufferl.load_policy(cfg, vecenv)  # if they support this

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
    vecenv = make_sudoku_vecenv("easy", num_envs=args.num_envs)
    algo.vecenv = vecenv  # depending on their API you may have to rebuild algo

    while algo.global_step < args.easy_steps:
        algo.evaluate()     # collect rollouts
        algo.train()        # do updates
        algo.print_dashboard()

    # === Medium phase ===
    vecenv.close()
    vecenv = make_sudoku_vecenv("medium", num_envs=args.num_envs)
    algo.vecenv = vecenv  # depending on their API you may have to rebuild algo

    while algo.global_step < args.medium_steps:
        algo.evaluate()
        algo.train()
        algo.print_dashboard()

    # === Hard phase ===
    vecenv.close()
    vecenv = make_sudoku_vecenv("hard", num_envs=args.num_envs)
    algo.vecenv = vecenv

    while algo.global_step < args.total_steps:
        algo.evaluate()
        algo.train()
        algo.print_dashboard()

    algo.save_checkpoint()
    vecenv.close()

if __name__ == "__main__":
    main()

