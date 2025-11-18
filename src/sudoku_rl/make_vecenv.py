# src/sudoku_rl/make_vecenv.py
import pufferlib.vector
from .env_puffer import SudokuPufferEnv

def make_sudoku_vecenv(difficulty: str, num_envs: int, seed: int = 0):
    return pufferlib.vector.make(
        SudokuPufferEnv,                      
        env_kwargs={"difficulty": difficulty},
        backend=pufferlib.vector.Serial,                     
        num_envs=num_envs,
        seed=seed,
    )
