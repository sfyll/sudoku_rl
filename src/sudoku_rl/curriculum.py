from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Sequence, Dict, Optional


@dataclass(frozen=True)
class BucketDef:
    """Minimal bucket descriptor.

    We keep only a stable id/bin_label so we don't duplicate metadata that
    already lives in the dataset manifest. The curriculum manager owns the
    lock state and statistics separately.
    """

    id: str
    bin_label: str


@dataclass
class EpisodeSummary:
    """Compact per-episode summary used for rolling stats."""

    solved: bool
    clean_solve: bool
    total_return: float          # scaled return (fed to learner/logs)
    total_return_raw: float      # unscaled MDP return (for scaling stats)
    length: int


class ReturnStats:
    """Online mean/std tracker (Welford) for raw episodic returns."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return (self.m2 / (self.count - 1)) ** 0.5


class BucketStats:
    """Rolling, windowed statistics for a single difficulty bucket."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.history: Deque[EpisodeSummary] = deque()
        self._solved = 0
        self._clean = 0
        self._return_sum = 0.0
        self._length_sum = 0

    # --- Properties ---
    @property
    def n(self) -> int:
        return len(self.history)

    @property
    def solve_rate(self) -> float:
        return self._solved / self.n if self.n else 0.0

    @property
    def clean_solve_rate(self) -> float:
        return self._clean / self.n if self.n else 0.0

    @property
    def avg_return(self) -> float:
        return self._return_sum / self.n if self.n else 0.0

    @property
    def avg_length(self) -> float:
        return self._length_sum / self.n if self.n else 0.0

    # --- Mutations ---
    def add(self, summary: EpisodeSummary) -> None:
        self.history.append(summary)
        self._solved += int(summary.solved)
        self._clean += int(summary.clean_solve)
        self._return_sum += summary.total_return
        self._length_sum += summary.length

        if self.n > self.window_size:
            old = self.history.popleft()
            self._solved -= int(old.solved)
            self._clean -= int(old.clean_solve)
            self._return_sum -= old.total_return
            self._length_sum -= old.length

    def to_logging_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}/solve_rate": self.solve_rate,
            f"{prefix}/clean_solve_rate": self.clean_solve_rate,
            f"{prefix}/avg_return": self.avg_return,
            f"{prefix}/avg_length": self.avg_length,
        }


class CurriculumManager:
    """Success-rate driven curriculum with promotion/down-weight rules.

    The manager is intentionally small: it owns bucket lock state, rolling
    stats, and sampling weights. All other logic (sampling puzzles, logging)
    is handled by callers to avoid duplication.
    """

    def __init__(
        self,
        bucket_defs: Sequence[BucketDef],
        *,
        initial_unlocked: int = 2,
        window_size: int = 200,
        promote_threshold: float = 0.70,
        promote_thresholds: Optional[Sequence[float]] = None,
        demote_threshold: float = 0.20,
        min_episodes_for_decision: int = 100,
        alpha: float = 2.0,
        eps: float = 0.05,
        underperforming_weight: float = 0.3,
        age_floor: float = 0.02,
        rng: random.Random | None = None,
        reward_target_std: float = 10.0,
        reward_scale_min: float = 0.1,
        reward_scale_max: float = 10.0,
        reward_eps: float = 1e-6,
        min_episodes_for_scale: int = 5,
    ) -> None:
        if initial_unlocked < 1:
            raise ValueError("At least one bucket must start unlocked")
        if len(bucket_defs) < initial_unlocked:
            raise ValueError("initial_unlocked cannot exceed number of buckets")

        self.bucket_defs: List[BucketDef] = list(bucket_defs)
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.promote_thresholds = list(promote_thresholds) if promote_thresholds is not None else None
        self.demote_threshold = demote_threshold
        self.min_episodes_for_decision = min_episodes_for_decision
        self.alpha = alpha
        self.eps = eps
        self.underperforming_weight = underperforming_weight
        self.age_floor = age_floor
        self.rng = rng or random.Random()
        # Reward scaling hyperparameters
        self.reward_target_std = reward_target_std
        self.reward_scale_min = reward_scale_min
        self.reward_scale_max = reward_scale_max
        self.reward_eps = reward_eps
        self.min_episodes_for_scale = min_episodes_for_scale

        # State
        self._locked: List[bool] = [True] * len(bucket_defs)
        for i in range(initial_unlocked):
            self._locked[i] = False
        self.max_unlocked_index = initial_unlocked - 1
        self.stats: List[BucketStats] = [BucketStats(window_size) for _ in bucket_defs]
        self.return_stats: List[ReturnStats] = [ReturnStats() for _ in bucket_defs]
        self._underperforming: set[int] = set()
        self.total_episodes = 0

    # --- Sampling ---
    def _clamped_solve_rate(self, idx: int) -> float:
        # Use clean solve rate as proficiency signal; raw solve_rate can be
        # misleading if the agent "solves" after many wrong digits.
        p = self.stats[idx].clean_solve_rate
        return max(self.eps, min(1 - self.eps, p))

    def _sampling_weight(self, idx: int) -> float:
        p = self._clamped_solve_rate(idx)
        age = max(self.age_floor, min(1.0, self.stats[idx].n / self.window_size))
        w = (1.0 - p) ** self.alpha
        w *= age
        if idx in self._underperforming:
            w *= self.underperforming_weight
        return w

    def choose_bucket(self) -> int:
        unlocked = [i for i, locked in enumerate(self._locked) if not locked]
        if not unlocked:
            unlocked = [0]

        weights = [self._sampling_weight(i) for i in unlocked]
        if not any(weights):
            weights = [1.0] * len(unlocked)

        return self.rng.choices(unlocked, weights=weights, k=1)[0]

    # --- Updates ---
    def update_after_episode(self, bucket_idx: int, summary: EpisodeSummary) -> None:
        self.stats[bucket_idx].add(summary)
        self.return_stats[bucket_idx].update(summary.total_return_raw)
        self.total_episodes += 1
        self._maybe_promote(bucket_idx)
        self._maybe_flag_underperforming()

    # --- Reward scaling for per-bucket normalization ---
    def get_scale(self, bucket_idx: int) -> float:
        stats = self.return_stats[bucket_idx]
        if stats.count < self.min_episodes_for_scale:
            return 1.0
        std = stats.std
        scale = self.reward_target_std / (std + self.reward_eps)
        return max(self.reward_scale_min, min(self.reward_scale_max, scale))

    def _maybe_promote(self, idx: int) -> None:
        next_idx = idx + 1
        if next_idx >= len(self.bucket_defs):
            return
        if not self._locked[next_idx]:
            return
        stats = self.stats[idx]
        if stats.n < self.min_episodes_for_decision:
            return
        threshold = self._threshold_for(idx)
        if stats.clean_solve_rate < threshold:
            return

        self._locked[next_idx] = False
        self.max_unlocked_index = max(self.max_unlocked_index, next_idx)

    def _maybe_flag_underperforming(self) -> None:
        j = self.max_unlocked_index
        stats = self.stats[j]
        if stats.n < self.min_episodes_for_decision:
            return
        if stats.clean_solve_rate <= self.demote_threshold:
            self._underperforming.add(j)
        else:
            self._underperforming.discard(j)

    def _threshold_for(self, idx: int) -> float:
        if self.promote_thresholds:
            if idx < len(self.promote_thresholds):
                return self.promote_thresholds[idx]
            return self.promote_thresholds[-1]
        return self.promote_threshold

    # --- Logging ---
    def metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "curriculum/max_unlocked_index": float(self.max_unlocked_index)
        }

        total_eps = sum(s.n for s in self.stats)
        total_solved = sum(s._solved for s in self.stats)
        if total_eps:
            out["curriculum/global_solve_rate"] = total_solved / total_eps

        for i, (bucket, stats) in enumerate(zip(self.bucket_defs, self.stats)):
            prefix = f"bucket_{i}_{bucket.id}"
            out.update(stats.to_logging_dict(prefix))
            out[f"{prefix}/sampling_weight"] = self._sampling_weight(i)
            rs = self.return_stats[i]
            out[f"{prefix}/return_raw_mean"] = rs.mean
            out[f"{prefix}/return_raw_std"] = rs.std
            out[f"{prefix}/reward_scale"] = self.get_scale(i)
        return out


def build_default_buckets(supported: Sequence[str], max_buckets: int = 6) -> List[BucketDef]:
    """Select a slim, monotonic subset of manifest bins for the curriculum.

    We keep the earliest `max_buckets` bins (easiest first). Caller should
    ensure `supported` is sorted by difficulty as in the manifest helper.
    """

    labels = list(supported)[:max_buckets]
    if len(labels) < 2:
        raise ValueError("Need at least two bins to build a curriculum")
    return [BucketDef(id=lbl, bin_label=lbl) for lbl in labels]
