# src/sudoku_rl/puzzles.py
from __future__ import annotations

import csv
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 0 or . = empty

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_FILES = {
    "tiny": DATA_DIR / "sudoku_tiny.csv",
    "very_easy": DATA_DIR / "sudoku_very_easy.csv",
    "easy": DATA_DIR / "sudoku_easy.csv",
    "moderate": DATA_DIR / "sudoku_moderate.csv",
    "medium": DATA_DIR / "sudoku_medium.csv",
    "tricky": DATA_DIR / "sudoku_tricky.csv",
    "hard": DATA_DIR / "sudoku_hard.csv",
}

SUPPORTED_DIFFICULTIES = tuple(SPLIT_FILES.keys())


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


def _normalize_difficulty(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in SUPPORTED_DIFFICULTIES:
        raise ValueError(
            f"Unknown difficulty '{label}'. Supported difficulties: {', '.join(SUPPORTED_DIFFICULTIES)}"
        )
    return normalized


def _load_from_csv(path: Path, difficulty: str) -> List[str]:
    puzzles: List[str] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "puzzle" not in reader.fieldnames:
            raise ValueError(f"CSV file {path} missing 'puzzle' column")
        for row in reader:
            puzzle = row["puzzle"].strip()
            if len(puzzle) != 81:
                raise ValueError(f"Puzzle '{puzzle}' in {path} must be length 81")
            puzzles.append(puzzle)
    if not puzzles:
        raise ValueError(f"No puzzles loaded for difficulty '{difficulty}' from {path}")
    return puzzles


@lru_cache(maxsize=1)
def _load_puzzle_pools() -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    for difficulty, csv_path in SPLIT_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV file not found: {csv_path}")
        pools[difficulty] = _load_from_csv(csv_path, difficulty)
    return pools


def get_puzzle_pool(difficulty: str) -> List[str]:
    normalized = _normalize_difficulty(difficulty)
    pools = _load_puzzle_pools()
    return pools[normalized]


def sample_puzzle(difficulty: str = "easy", seed: Optional[int] = None) -> np.ndarray:
    """Sample a random puzzle of given difficulty as a (9, 9) board."""
    pool = get_puzzle_pool(difficulty)
    if seed is not None:
        return _string_to_board(pool[seed])
    s = random.choice(pool)
    return _string_to_board(s)
