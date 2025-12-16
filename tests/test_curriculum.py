import random
from collections import Counter

import pytest

from sudoku_rl.curriculum import BucketDef, CurriculumManager, EpisodeSummary, BucketStats


def _summary(
    solved: bool,
    clean: bool = True,
    ret: float = 1.0,
    length: int = 5,
    wrongs: int = 0,
    empties: int = 5,
    start_F: float = 10.0,
) -> EpisodeSummary:
    return EpisodeSummary(
        solved=solved,
        clean_solve=clean,
        wrong_digit_count=wrongs,
        initial_empties=empties,
        start_F=start_F,
        total_return=ret,
        total_return_raw=ret,
        length=length,
    )


def test_bucket_stats_window_eviction():
    stats = BucketStats(window_size=2)
    stats.add(_summary(True, True, 2.0, 10))
    stats.add(_summary(False, False, -1.0, 8, wrongs=1, empties=4, start_F=8.0))

    assert stats.n == 2
    assert pytest.approx(stats.solve_rate, rel=1e-6) == 0.5
    assert pytest.approx(stats.avg_return, rel=1e-6) == 0.5

    # Third insert should evict the first
    stats.add(_summary(True, True, 3.0, 12, wrongs=0, empties=6, start_F=12.0))
    assert stats.n == 2
    assert pytest.approx(stats.solve_rate, rel=1e-6) == 0.5  # one solved, one not
    assert pytest.approx(stats.avg_return, rel=1e-6) == 1.0  # (-1 + 3) / 2
    assert stats.wrong_digit_rate == 0.5
    assert stats.steps_per_empty > 0


def test_promotion_unlocks_next_bucket():
    buckets = [BucketDef("b0", "b0"), BucketDef("b1", "b1")]
    mgr = CurriculumManager(
        buckets,
        initial_unlocked=1,
        window_size=5,
        min_episodes_for_decision=3,
        solve_threshold=0.8,
        clean_solve_threshold=0.6,
        wrong_digit_threshold=0.5,
        steps_per_empty_threshold=2.0,
        patience=1,
        rng=random.Random(0),
    )

    # Three episodes meet thresholds -> promotion
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))
    assert mgr.max_unlocked_index == 1

    # Patience should gate promotion
    mgr2 = CurriculumManager(
        buckets,
        initial_unlocked=1,
        window_size=5,
        min_episodes_for_decision=2,
        solve_threshold=1.0,
        clean_solve_threshold=1.0,
        wrong_digit_threshold=0.0,
        steps_per_empty_threshold=2.0,
        patience=2,
        rng=random.Random(0),
    )
    mgr2.update_after_episode(0, _summary(True, clean=True))
    mgr2.update_after_episode(0, _summary(True, clean=True))
    assert mgr2.max_unlocked_index == 0  # streak=1
    mgr2.update_after_episode(0, _summary(True, clean=True))
    assert mgr2.max_unlocked_index == 1  # streak hits 2


def test_sampling_mixture():
    buckets = [BucketDef("b0", "b0"), BucketDef("b1", "b1"), BucketDef("b2", "b2")]
    mgr = CurriculumManager(
        buckets,
        initial_unlocked=3,
        window_size=10,
        rng=random.Random(0),
    )
    mgr.max_unlocked_index = 2  # frontier
    draws = Counter(mgr.choose_bucket() for _ in range(1000))
    # Expected proportions: frontier ~60%, frontier-1 ~25%, others ~15%
    assert draws[2] > draws[1] > draws[0]
