# src/sudoku_rl/puzzles.py
from __future__ import annotations

import csv
import json
import random
import re
import time
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# 0 or . = empty

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = DATA_DIR / "sudoku_bins_manifest.json"
BIN_GLOB = "sudoku_zeros_*.csv"
PUZZLE_CACHE_PATH = DATA_DIR / "puzzle_cache.pkl"



def _string_to_board(s: str) -> np.ndarray:
    """81-char string -> (9, 9) np.ndarray[int], 0 = empty."""
    assert len(s) == 81, f"Puzzle string must be length 81, got {len(s)}"
    flat = []
    for ch in s:
        if ch in ("0", "."):
            flat.append(0)
        else:
            flat.append(int(ch))
    return np.array(flat, dtype=np.int8).reshape(9, 9)


def _parse_bin_label(label: str) -> Tuple[int, int]:
    """Parse labels like 'zeros_08_11' -> (8, 11)."""
    match = re.match(r"zeros_(\d{2})_(\d{2})", label)
    if not match:
        raise ValueError(f"Invalid bin label '{label}'. Expected format zeros_XX_YY.")
    lo, hi = int(match.group(1)), int(match.group(2))
    if lo > hi:
        raise ValueError(f"Bin label '{label}' has lo > hi")
    return lo, hi


def _load_manifest() -> List[Dict]:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open() as f:
            return json.load(f)
    return []


def _discover_bins_from_files() -> Dict[str, Path]:
    bins: Dict[str, Path] = {}
    for path in DATA_DIR.glob(BIN_GLOB):
        label = path.stem.replace("sudoku_", "")
        bins[label] = path
    return bins


@lru_cache(maxsize=1)
def _bin_index() -> Dict[str, Path]:
    manifest = _load_manifest()
    if manifest:
        return {entry["label"]: DATA_DIR / entry["path"] for entry in manifest}
    # Fallback: discover by glob if manifest is missing
    discovered = _discover_bins_from_files()
    if not discovered:
        raise FileNotFoundError(
            f"No sudoku bin files found. Run scripts/create_filtered_dataset_sudoku.py to generate them."
        )
    return discovered


@lru_cache(maxsize=1)
def _bin_rows() -> Dict[str, int]:
    manifest = _load_manifest()
    if manifest:
        return {entry["label"]: int(entry.get("rows", 0)) for entry in manifest}
    # If no manifest, we don't know row counts; assume >0 for discovered bins
    return {label: -1 for label in _discover_bins_from_files()}


@lru_cache(maxsize=1)
def supported_bins() -> Tuple[str, ...]:
    rows = _bin_rows()
    labels_with_data = [label for label, count in rows.items() if count != 0]
    return tuple(sorted(labels_with_data))


def _normalize_label(label: str) -> str:
    label = label.strip().lower().replace("-", "_").replace(" ", "_")
    if label in supported_bins():
        return label
    # Allow users to pass a raw integer as string
    if label.isdigit():
        return _nearest_bin_for_target(int(label))
    raise ValueError(
        f"Unknown bin '{label}'. Supported bins: {', '.join(supported_bins())} or pass target_zeros as int."
    )


def _nearest_bin_for_target(target_zeros: int) -> str:
    best_label = None
    best_dist = 1e9
    for label in supported_bins():
        lo, hi = _parse_bin_label(label)
        if lo <= target_zeros <= hi:
            return label
        center = (lo + hi) / 2
        dist = abs(center - target_zeros)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    if best_label is None:
        raise ValueError("No bins available to satisfy the request")
    return best_label


def _load_from_csv(path: Path, label: str) -> List[dict]:
    t0 = time.time()
    puzzles: List[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "puzzle" not in reader.fieldnames:
            raise ValueError(f"CSV file {path} missing 'puzzle' column")
        for row in reader:
            puzzle = row["puzzle"].strip()
            if len(puzzle) != 81:
                raise ValueError(f"Puzzle '{puzzle}' in {path} must be length 81")
            entry = {"puzzle": puzzle}
            if "solution" in row and row["solution"]:
                solution = row["solution"].strip()
                if len(solution) == 81:
                    entry["solution"] = solution
            if "zeros" in row:
                try:
                    entry["zeros"] = int(row["zeros"])
                except ValueError:
                    pass
            puzzles.append(entry)
    if not puzzles:
        raise ValueError(f"No puzzles loaded for bin '{label}' from {path}")
    print(f"[pid {os.getpid()}] loaded {len(puzzles)} puzzles for {label} from CSV in {time.time()-t0:.2f}s", flush=True)
    return puzzles


@lru_cache(maxsize=1)
def _load_puzzle_pools() -> Dict[str, List[dict]]:
    t0 = time.time()
    if PUZZLE_CACHE_PATH.exists():
        with PUZZLE_CACHE_PATH.open("rb") as f:
            pools = pickle.load(f)
        print(f"[pid {os.getpid()}] loaded puzzle cache from {PUZZLE_CACHE_PATH} in {time.time()-t0:.2f}s", flush=True)
        return pools

    pools: Dict[str, List[dict]] = {}
    rows = _bin_rows()
    for label, csv_path in _bin_index().items():
        if rows.get(label, -1) == 0:
            continue  # skip empty bins
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV file not found: {csv_path}")
        pools[label] = _load_from_csv(csv_path, label)

    with PUZZLE_CACHE_PATH.open("wb") as f:
        pickle.dump(pools, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[pid {os.getpid()}] built puzzle cache at {PUZZLE_CACHE_PATH} in {time.time()-t0:.2f}s", flush=True)
    return pools


def get_puzzle_pool(label: str) -> List[dict]:
    normalized = _normalize_label(label)
    pools = _load_puzzle_pools()
    return pools[normalized]


def sample_puzzle(
    bin_label: str | None = None,
    *,
    target_zeros: Optional[int] = None,
    seed: Optional[int] = None,
    return_solution: bool = False,
    prev_mix_ratio: float = 0.3,
):
    """
    Sample a random puzzle of a given bin.

    Args:
        bin_label: canonical label like "zeros_08_11". If None, uses difficulty/target_zeros.
        difficulty: legacy alias; will be mapped to the nearest bin. Kept for backward compatibility.
        target_zeros: integer target; nearest bin is selected.
        seed: optional deterministic index.
        return_solution: if True, returns (board, solution).
        prev_mix_ratio: probability of sampling from an earlier (easier) bin to reduce catastrophic forgetting.
    """
    if bin_label is None:
        if target_zeros is not None:
            bin_label = _nearest_bin_for_target(int(target_zeros))
        else:
            # default to easiest bin
            bin_label = supported_bins()[0]
    else:
        bin_label = _normalize_label(bin_label)

    # Optionally mix in puzzles from previous bins (uniform over them).
    def _choose_bin(label: str) -> str:
        if prev_mix_ratio <= 0:
            return label

        bins = list(supported_bins())
        if label not in bins:
            return label

        idx = bins.index(label)
        if idx == 0:
            return label  # no easier bins

        # Filter out bins with no rows (defensive: manifest may list zero-row bins)
        previous_bins = []
        for b in bins[:idx]:
            try:
                if len(get_puzzle_pool(b)) > 0:
                    previous_bins.append(b)
            except (ValueError, FileNotFoundError):
                # Skip bins that failed to load
                continue
        if not previous_bins:
            return label

        # Draw from previous bins with the configured probability.
        if random.random() < prev_mix_ratio:
            return random.choice(previous_bins)
        return label

    selected_bin = _choose_bin(bin_label)

    pool = get_puzzle_pool(selected_bin)
    # Use seed for reproducible sampling without exceeding pool length.
    if seed is None:
        idx = random.randrange(len(pool))
    else:
        idx = seed % len(pool)
    row = pool[idx]
    puzzle_str = row["puzzle"] if isinstance(row, dict) else row
    solution_str = row.get("solution") if isinstance(row, dict) else None

    puzzle_board = _string_to_board(puzzle_str)
    if not return_solution:
        return puzzle_board

    if solution_str is None:
        raise ValueError(f"Solution not available for bin {bin_label}")
    solution_board = _string_to_board(solution_str)
    return puzzle_board, solution_board
