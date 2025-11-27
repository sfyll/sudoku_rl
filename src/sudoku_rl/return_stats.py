import multiprocessing as mp
from typing import Sequence

class SharedReturnStatsRegistry:
    """Process-safe shared return stats for per-bucket reward scaling.

    Uses Welford updates under a single lock. Scales are global across envs.
    """

    def __init__(
        self,
        num_buckets: int,
        *,
        target_std: float = 5.0,
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        eps: float = 1e-6,
        min_episodes_for_scale: int = 50,
    ) -> None:
        manager = mp.Manager()
        self._manager = manager  # keep alive in the creating process
        self.count = manager.list([0] * num_buckets)
        self.mean = manager.list([0.0] * num_buckets)
        self.m2 = manager.list([0.0] * num_buckets)
        self.lock = manager.Lock()
        self.target_std = target_std
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.eps = eps
        self.min_episodes_for_scale = min_episodes_for_scale

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_manager", None)  # manager itself isn't picklable
        return state

    def update(self, bucket_idx: int, x: float) -> None:
        with self.lock:
            c = self.count[bucket_idx] + 1
            delta = x - self.mean[bucket_idx]
            mean_new = self.mean[bucket_idx] + delta / c
            delta2 = x - mean_new
            m2_new = self.m2[bucket_idx] + delta * delta2

            self.count[bucket_idx] = c
            self.mean[bucket_idx] = mean_new
            self.m2[bucket_idx] = m2_new

    def _std(self, bucket_idx: int) -> float:
        c = self.count[bucket_idx]
        if c < 2:
            return 0.0
        return (self.m2[bucket_idx] / (c - 1)) ** 0.5

    def get_scale(self, bucket_idx: int) -> float:
        with self.lock:
            c = self.count[bucket_idx]
            if c < self.min_episodes_for_scale:
                return 1.0
            std = self._std(bucket_idx)
        scale = self.target_std / (std + self.eps)
        if scale < self.scale_min:
            scale = self.scale_min
        elif scale > self.scale_max:
            scale = self.scale_max
        return scale

    def summary(self) -> list[dict]:
        with self.lock:
            out = []
            for i in range(len(self.count)):
                std = self._std(i)
                out.append(
                    {
                        "count": self.count[i],
                        "mean": self.mean[i],
                        "std": std,
                        "scale": self.target_std / (std + self.eps) if self.count[i] >= self.min_episodes_for_scale else 1.0,
                    }
                )
        return out
