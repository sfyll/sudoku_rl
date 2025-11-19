# src/sudoku_rl/make_vecenv.py
import pufferlib.vector
from .env_puffer import SudokuPufferEnv

def make_sudoku_vecenv(
    difficulty: str,
    num_envs: int,
    seed: int = 0,
    max_steps: int | None = None,
):
    env_kwargs = {"difficulty": difficulty}
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps

    return pufferlib.vector.make(
        SudokuPufferEnv,
        env_kwargs=env_kwargs,
        backend=pufferlib.vector.Serial,
        num_envs=num_envs,
        seed=seed,
    )
