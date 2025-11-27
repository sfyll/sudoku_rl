import multiprocessing as mp


class SharedReturnStatsRegistry:
    """Process-safe shared return stats for per-bucket reward scaling.

    Previous version used ``multiprocessing.Manager`` proxies which incurred a
    heavy IPC round-trip on every call. This version keeps the same API but
    stores the accumulators in shared memory (``mp.Array``) and guards updates
    with a single ``mp.Lock``. No manager process is spawned, so ``get_scale``
    and ``update`` are just shared-memory reads/writes.
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
        # Shared memory buffers (no manager proxies)
        # 'q' -> signed long long for counts, 'd' -> double for means/M2
        self.count = mp.Array("q", [0] * num_buckets, lock=False)
        self.mean = mp.Array("d", [0.0] * num_buckets, lock=False)
        self.m2 = mp.Array("d", [0.0] * num_buckets, lock=False)

        # Single process-shared lock to keep Welford updates atomic
        self.lock = mp.Lock()

        self.target_std = target_std
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.eps = eps
        self.min_episodes_for_scale = min_episodes_for_scale

    def update(self, bucket_idx: int, x: float) -> None:
        with self.lock:
            c = self.count[bucket_idx] + 1
            mean_prev = self.mean[bucket_idx]
            delta = x - mean_prev
            mean_new = mean_prev + delta / c
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
