# src/sudoku_rl/make_vecenv.py
from typing import Optional
import pufferlib.vector
from .env_puffer import SudokuPufferEnv


def make_sudoku_vecenv(
    bin_label: str,
    num_envs: int,
    seed: Optional[int] = None,
    max_steps: int | None = None,
    backend=None,
    num_workers: int | None = None,
    terminate_on_wrong_digit: bool = True,
    prev_mix_ratio: float = 0.3,
    bucket_defs=None,
    curriculum_kwargs=None,
):
    """Create a Puffer vecenv with optional threaded/multiprocess backend."""

    env_kwargs = {
        "bin_label": bin_label,
        "terminate_on_wrong_digit": terminate_on_wrong_digit,
        "prev_mix_ratio": prev_mix_ratio,
    }
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    if bucket_defs is not None:
        env_kwargs["bucket_defs"] = bucket_defs
    if curriculum_kwargs is not None:
        env_kwargs["curriculum_kwargs"] = curriculum_kwargs

    backend = backend or pufferlib.vector.Multiprocessing
    kwargs = dict(
        backend=backend,
        num_envs=num_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
    if num_workers is not None:
        kwargs["num_workers"] = num_workers

    return pufferlib.vector.make(SudokuPufferEnv, **kwargs)
