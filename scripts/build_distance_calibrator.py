#!/usr/bin/env python
"""
Fit isotonic distance calibrator h over solver traces and save to experiments/distance_calibrator.json.

This is the CLI companion to the notebook calibration section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sudoku_rl.distance_regressor import IsotonicCalibrator, DistanceRegressor
from sudoku_rl.solver_norvig import solve_with_trace
from sudoku_rl.puzzle import sample_puzzle, supported_bins


def collect_samples(model: DistanceRegressor, bins, traces_per_bin: int, max_states: int, device) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    total = 0
    for b in bins:
        for seed in range(traces_per_bin):
            puzzle, _ = sample_puzzle(bin_label=b, seed=seed, return_solution=True, prev_mix_ratio=0.0)
            trace = solve_with_trace(puzzle)
            if trace is None:
                continue
            boards = np.stack([t.reshape(-1) for t in trace]).astype(np.float32) / 9.0
            with torch.no_grad():
                preds = model(torch.as_tensor(boards, device=device)).cpu().numpy().reshape(-1)
            true_d = np.arange(len(trace) - 1, -1, -1, dtype=np.float64)
            xs.append(preds)
            ys.append(true_d)
            total += len(preds)
            if total >= max_states:
                break
        if total >= max_states:
            break
    if not xs:
        raise RuntimeError("No calibration samples collected")
    return np.concatenate(xs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("experiments/distance_regressor.pt"))
    parser.add_argument("--out", type=Path, default=Path("experiments/distance_calibrator.json"))
    parser.add_argument("--bins", nargs="*", default=list(supported_bins()))
    parser.add_argument("--traces-per-bin", type=int, default=60)
    parser.add_argument("--max-states", type=int, default=200_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model, map_location=device)
    model = DistanceRegressor().to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    x, y = collect_samples(model, args.bins, args.traces_per_bin, args.max_states, device)
    calib = IsotonicCalibrator().fit(x, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calib.to_dict(), indent=2))
    print(f"Saved calibrator with {len(x)} samples to {args.out}")


if __name__ == "__main__":
    main()
