import random
from collections import Counter

import pytest

from sudoku_rl.curriculum import BucketDef, CurriculumManager, EpisodeSummary, BucketStats


def _summary(solved: bool, clean: bool = True, ret: float = 1.0, length: int = 5) -> EpisodeSummary:
    return EpisodeSummary(solved=solved, clean_solve=clean, total_return=ret, length=length)


def test_bucket_stats_window_eviction():
    stats = BucketStats(window_size=2)
    stats.add(_summary(True, True, 2.0, 10))
    stats.add(_summary(False, False, -1.0, 8))

    assert stats.n == 2
    assert pytest.approx(stats.solve_rate, rel=1e-6) == 0.5
    assert pytest.approx(stats.avg_return, rel=1e-6) == 0.5

    # Third insert should evict the first
    stats.add(_summary(True, True, 3.0, 12))
    assert stats.n == 2
    assert pytest.approx(stats.solve_rate, rel=1e-6) == 0.5  # one solved, one not
    assert pytest.approx(stats.avg_return, rel=1e-6) == 1.0  # (-1 + 3) / 2


def test_promotion_unlocks_next_bucket():
    buckets = [BucketDef("b0", "b0"), BucketDef("b1", "b1")]
    mgr = CurriculumManager(
        buckets,
        initial_unlocked=1,
        window_size=5,
        promote_threshold=0.6,
        min_episodes_for_decision=3,
        rng=random.Random(0),
    )

    # Mixed solves but only 2 clean solves -> no promotion yet
    mgr.update_after_episode(0, _summary(True, clean=False))
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))

    assert mgr.max_unlocked_index == 1

    # If clean rate dipped, promotion should not happen
    mgr2 = CurriculumManager(
        buckets,
        initial_unlocked=1,
        window_size=5,
        promote_threshold=0.7,
        min_episodes_for_decision=3,
        rng=random.Random(0),
    )
    mgr2.update_after_episode(0, _summary(True, clean=False))
    mgr2.update_after_episode(0, _summary(True, clean=False))
    mgr2.update_after_episode(0, _summary(True, clean=True))
    assert mgr2.max_unlocked_index == 0


def test_per_bucket_thresholds():
    buckets = [BucketDef("b0", "b0"), BucketDef("b1", "b1"), BucketDef("b2", "b2"), BucketDef("b3", "b3")]
    mgr = CurriculumManager(
        buckets,
        initial_unlocked=1,
        window_size=5,
        promote_thresholds=[0.9, 0.8, 0.6],
        min_episodes_for_decision=3,
        rng=random.Random(0),
    )

    # Bucket0 requires 0.9
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(False, clean=False))  # clean rate 0.67 -> no promotion
    assert mgr.max_unlocked_index == 0

    mgr.update_after_episode(0, _summary(True, clean=True))  # clean rate now 0.75 -> still below 0.9
    assert mgr.max_unlocked_index == 0

    # push clean rate to 1.0 over window 5 (evict the earlier false)
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))
    mgr.update_after_episode(0, _summary(True, clean=True))
    assert mgr.max_unlocked_index == 1  # promoted to bucket1

    # Bucket1 requires 0.8
    mgr.update_after_episode(1, _summary(True, clean=True))
    mgr.update_after_episode(1, _summary(True, clean=True))
    mgr.update_after_episode(1, _summary(False, clean=False))  # clean rate 0.67 -> no promotion
    assert mgr.max_unlocked_index == 1

    mgr.update_after_episode(1, _summary(True, clean=True))  # clean rate 0.75 -> still below 0.8
    mgr.update_after_episode(1, _summary(True, clean=True))  # clean rate 0.8 -> promote to bucket2
    assert mgr.max_unlocked_index == 2


def test_sampling_prefers_mid_proficiency():
    buckets = [BucketDef("easy", "easy"), BucketDef("mid", "mid")]
    mgr = CurriculumManager(
        buckets,
        initial_unlocked=2,
        window_size=200,
        rng=random.Random(0),
    )

    # Bucket 0 mastered
    for _ in range(120):
        mgr.update_after_episode(0, _summary(True, ret=2.0))
    # Bucket 1 around 50% solve rate
    for _ in range(60):
        mgr.update_after_episode(1, _summary(True, ret=1.5))
    for _ in range(60):
        mgr.update_after_episode(1, _summary(False, clean=False, ret=-1.0))

    draws = Counter(mgr.choose_bucket() for _ in range(500))
    assert draws[1] > draws[0]  # prefers the mid bucket over mastered


def test_underperforming_bucket_downweighted():
    buckets = [BucketDef("easy", "easy"), BucketDef("hard", "hard")]
    mgr = CurriculumManager(
        buckets,
        initial_unlocked=2,
        window_size=50,
        min_episodes_for_decision=5,
        demote_threshold=0.2,
        underperforming_weight=0.1,
        rng=random.Random(1),
    )

    # Hard bucket performs poorly
    for _ in range(5):
        mgr.update_after_episode(1, _summary(False, clean=False, ret=-2.0))
    # Easy bucket has good rate
    for _ in range(5):
        mgr.update_after_episode(0, _summary(True, ret=2.0))

    baseline_hard = (1 - mgr.eps) ** mgr.alpha
    weight_hard = mgr._sampling_weight(1)
    assert weight_hard < baseline_hard  # down-weight applied
