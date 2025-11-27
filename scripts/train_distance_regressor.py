#!/usr/bin/env python
"""Train distance-to-solve regressor using deterministic solver traces."""
from __future__ import annotations
import argparse
from pathlib import Path
import random

import numpy as np
import torch

from sudoku_rl.distance_regressor import (
    DistanceDataset,
    generate_distance_examples,
    train_regressor,
    evaluate,
)
from sudoku_rl.puzzle import supported_bins


def split_ids(unique_ids: np.ndarray, seed: int, val_frac: float = 0.1, test_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_ids)
    n = len(perm)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = perm[:n_test]
    val_ids = perm[n_test:n_test + n_val]
    train_ids = perm[n_test + n_val:]
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", nargs="*", default=["zeros_12_15", "zeros_24_27", "zeros_32_35", "zeros_44_47", "zeros_48_51"], help="Bin labels to sample")
    parser.add_argument("--puzzles-per-bin", type=int, default=-1, help="Base puzzles per bin (prod defaults 5000/2000; dev 1000/500). Set <=0 to use defaults.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("experiments/test_distance_regressor.pt"))
    parser.add_argument("--prod", action="store_true", help="Full run: all bins, ~5k per bin (2k for hardest); bump epochs.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.prod:
        args.bins = list(supported_bins())
        base = 5000 if args.puzzles_per_bin <= 0 else args.puzzles_per_bin
        hard = min(base, 2000)
        args.epochs = max(args.epochs, 10)
        print(f"[prod] bins={args.bins}; base={base}, hard={hard}; epochs={args.epochs}")
        if args.out == Path("experiments/test_distance_regressor.pt"):
            args.out = Path("experiments/distance_regressor.pt")
    else:
        for b in args.bins:
            if b not in supported_bins():
                raise ValueError(f"Unknown bin {b}; available: {supported_bins()}")
        base = 1000 if args.puzzles_per_bin <= 0 else args.puzzles_per_bin
        hard = min(base, 500)
        print(f"[dev] bins={args.bins}; base={base}, hard={hard}; epochs={args.epochs}")

    def _hi(bin_label: str) -> int:
        parts = bin_label.split("_")
        try:
            return int(parts[-1])
        except Exception:
            return 0

    per_bin_counts = {b: (hard if _hi(b) > 47 else base) for b in args.bins}

    print("Generating solver traces...")
    print("Sampling plan per bin:")
    for b in args.bins:
        print(f"  {b}: {per_bin_counts[b]}")

    examples = generate_distance_examples(args.bins, puzzles_per_bin=per_bin_counts, seed_offset=args.seed)
    print(f"Examples: {len(examples)}")

    puzzle_ids = np.array([ex.puzzle_id for ex in examples])
    unique_ids = np.unique(puzzle_ids)
    train_ids, val_ids, test_ids = split_ids(unique_ids, seed=args.seed)

    def select(ids):
        id_set = set(ids.tolist())
        return [ex for ex in examples if ex.puzzle_id in id_set]

    train_ds = DistanceDataset(select(train_ids))
    val_ds = DistanceDataset(select(val_ids))
    test_ds = DistanceDataset(select(test_ids))

    print(f"Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")

    model, history = train_regressor(train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, verbose=True)
    metrics = evaluate(model, test_ds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "bins": args.bins,
        "puzzles_per_bin": per_bin_counts,
        "history": history,
        "metrics": metrics,
        "seed": args.seed,
    }, args.out)

    print("Saved", args.out)
    print("Test metrics", metrics)


if __name__ == "__main__":
    main()
