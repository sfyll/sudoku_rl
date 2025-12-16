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
    wrong_digit_count: int
    initial_empties: int
    start_F: float
    total_return: float          # scaled return (fed to learner/logs)
    total_return_raw: float      # unscaled MDP return (for external scaling stats)
    length: int


class BucketStats:
    """Rolling, windowed statistics for a single difficulty bucket."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.history: Deque[EpisodeSummary] = deque()
        self._solved = 0
        self._clean = 0
        self._wrong_episodes = 0
        self._return_sum = 0.0
        self._length_sum = 0
        self._steps_per_empty_sum = 0.0
        self._start_F_sum = 0.0

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

    @property
    def wrong_digit_rate(self) -> float:
        return self._wrong_episodes / self.n if self.n else 0.0

    @property
    def steps_per_empty(self) -> float:
        return self._steps_per_empty_sum / self.n if self.n else 0.0

    @property
    def start_F_mean(self) -> float:
        return self._start_F_sum / self.n if self.n else 0.0

    # --- Mutations ---
    def add(self, summary: EpisodeSummary) -> None:
        self.history.append(summary)
        self._solved += int(summary.solved)
        self._clean += int(summary.clean_solve)
        self._wrong_episodes += int(summary.wrong_digit_count > 0)
        self._return_sum += summary.total_return
        self._length_sum += summary.length
        self._steps_per_empty_sum += summary.length / max(1, summary.initial_empties)
        self._start_F_sum += summary.start_F

        if self.n > self.window_size:
            old = self.history.popleft()
            self._solved -= int(old.solved)
            self._clean -= int(old.clean_solve)
            self._wrong_episodes -= int(old.wrong_digit_count > 0)
            self._return_sum -= old.total_return
            self._length_sum -= old.length
            self._steps_per_empty_sum -= old.length / max(1, old.initial_empties)
            self._start_F_sum -= old.start_F

    def to_logging_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}/solve_rate": self.solve_rate,
            f"{prefix}/clean_solve_rate": self.clean_solve_rate,
            f"{prefix}/avg_return": self.avg_return,
            f"{prefix}/avg_length": self.avg_length,
            f"{prefix}/wrong_digit_rate": self.wrong_digit_rate,
            f"{prefix}/steps_per_empty": self.steps_per_empty,
            f"{prefix}/start_F_mean": self.start_F_mean,
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
        min_episodes_for_decision: int = 100,
        # Frontier unlock thresholds
        solve_threshold: float = 0.95,
        clean_solve_threshold: float = 0.60,
        wrong_digit_threshold: float = 0.10,
        steps_per_empty_threshold: float = 1.5,
        patience: int = 3,
        rng: random.Random | None = None,
    ) -> None:
        if initial_unlocked < 1:
            raise ValueError("At least one bucket must start unlocked")
        if len(bucket_defs) < initial_unlocked:
            raise ValueError("initial_unlocked cannot exceed number of buckets")

        self.bucket_defs: List[BucketDef] = list(bucket_defs)
        self.window_size = window_size
        self.min_episodes_for_decision = min_episodes_for_decision
        self.solve_threshold = solve_threshold
        self.clean_solve_threshold = clean_solve_threshold
        self.wrong_digit_threshold = wrong_digit_threshold
        self.steps_per_empty_threshold = steps_per_empty_threshold
        self.patience = patience
        self.rng = rng or random.Random()

        # State
        self._locked: List[bool] = [True] * len(bucket_defs)
        for i in range(initial_unlocked):
            self._locked[i] = False
        self.max_unlocked_index = initial_unlocked - 1
        self.stats: List[BucketStats] = [BucketStats(window_size) for _ in bucket_defs]
        self.total_episodes = 0
        self._frontier_streak = 0

    # --- Sampling ---
    def choose_bucket(self) -> int:
        k = self.max_unlocked_index
        if k <= 0:
            return 0

        r = self.rng.random()
        if r < 0.60:
            return k
        elif r < 0.85:
            return k - 1
        else:
            if k - 1 <= 0:
                return 0
            return self.rng.randint(0, k - 2)

    # --- Updates ---
    def update_after_episode(self, bucket_idx: int, summary: EpisodeSummary) -> None:
        self.stats[bucket_idx].add(summary)
        self.total_episodes += 1
        if bucket_idx == self.max_unlocked_index:
            self._maybe_promote_frontier()

    def _frontier_metrics(self) -> Dict[str, float]:
        idx = self.max_unlocked_index
        stats = self.stats[idx]
        return {
            "solve_rate": stats.solve_rate,
            "clean_solve_rate": stats.clean_solve_rate,
            "wrong_digit_rate": stats.wrong_digit_rate,
            "steps_per_empty": stats.steps_per_empty,
            "start_F_mean": stats.start_F_mean,
            "episodes": float(stats.n),
        }

    def _maybe_promote_frontier(self) -> None:
        idx = self.max_unlocked_index
        next_idx = idx + 1
        if next_idx >= len(self.bucket_defs):
            return
        if not self._locked[next_idx]:
            return
        stats = self.stats[idx]
        if stats.n < self.min_episodes_for_decision:
            return
        if (
            stats.solve_rate >= self.solve_threshold
            and stats.clean_solve_rate >= self.clean_solve_threshold
            and stats.wrong_digit_rate <= self.wrong_digit_threshold
            and stats.steps_per_empty <= self.steps_per_empty_threshold
        ):
            self._frontier_streak += 1
        else:
            self._frontier_streak = 0

        if self._frontier_streak < self.patience:
            return

        self._frontier_streak = 0
        self._locked[next_idx] = False
        self.max_unlocked_index = max(self.max_unlocked_index, next_idx)

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

            # Mirror key bin-level stats under env/ for easier TensorBoard filtering
            env_prefix = f"env/bin_{i}"
            out[f"{env_prefix}/solve_rate"] = stats.solve_rate
            out[f"{env_prefix}/clean_solve_rate"] = stats.clean_solve_rate
            out[f"{env_prefix}/wrong_digit_rate"] = stats.wrong_digit_rate
            out[f"{env_prefix}/steps_per_empty"] = stats.steps_per_empty
            out[f"{env_prefix}/start_F_mean"] = stats.start_F_mean
            out[f"{env_prefix}/episodes"] = float(stats.n)

        # Frontier-only diagnostics
        frontier = self._frontier_metrics()
        out.update({
            "env/frontier_solve_rate": frontier["solve_rate"],
            "env/frontier_clean_solve_rate": frontier["clean_solve_rate"],
            "env/frontier_wrong_digit_rate": frontier["wrong_digit_rate"],
            "env/frontier_steps_per_empty": frontier["steps_per_empty"],
            "env/frontier_start_F_mean": frontier["start_F_mean"],
        })
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
