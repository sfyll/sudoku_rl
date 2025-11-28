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
    prev_mix_ratio: float = 0.3,
    bucket_defs=None,
    curriculum_kwargs=None,
    shared_return_stats=None,
    vec_batch_size: int | None = None,
    vec_zero_copy: bool = False,
    vec_overwork: bool = False,
):
    """Create a Puffer vecenv with optional threaded/multiprocess backend."""

    env_kwargs = {
        "bin_label": bin_label,
        "prev_mix_ratio": prev_mix_ratio,
    }
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    if bucket_defs is not None:
        env_kwargs["bucket_defs"] = bucket_defs
    if curriculum_kwargs is not None:
        env_kwargs["curriculum_kwargs"] = curriculum_kwargs
    else:
        # single-bin, no curriculum; keep parameters compatible with CurriculumManager signature
        env_kwargs["curriculum_kwargs"] = {
            "initial_unlocked": 1,
            "window_size": 200,
            "promote_threshold": 1.0,
            "min_episodes_for_decision": 1,
        }
    if shared_return_stats is not None:
        env_kwargs["shared_return_stats"] = shared_return_stats

    backend = backend or pufferlib.vector.Multiprocessing
    kwargs = dict(
        backend=backend,
        num_envs=num_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )
    if num_workers is not None:
        kwargs["num_workers"] = num_workers

    # Multiprocessing backend: tune batch/sync behavior to reduce recv stalls
    if backend is pufferlib.vector.Multiprocessing:
        if vec_batch_size is not None:
            kwargs["batch_size"] = vec_batch_size
        kwargs["zero_copy"] = vec_zero_copy
        kwargs["overwork"] = vec_overwork

    return pufferlib.vector.make(SudokuPufferEnv, **kwargs)
