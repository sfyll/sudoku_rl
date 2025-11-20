import pandas as pd
from pathlib import Path
import numpy as np


INPUT_PATH = Path("data/sudoku.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. Load raw Kaggle data ----
df = pd.read_csv(INPUT_PATH)

# Try to detect puzzle and solution columns
if "puzzle" in df.columns:
    puzzle_col = "puzzle"
elif "quizzes" in df.columns:
    puzzle_col = "quizzes"
else:
    raise ValueError(f"Could not find puzzle column in {df.columns}")

if "solution" in df.columns:
    solution_col = "solution"
elif "solutions" in df.columns:
    solution_col = "solutions"
else:
    raise ValueError(f"Could not find solution column in {df.columns}")

print(f"Puzzle Loaded, {df.shape[0]} rows")
# ---- 2. Compute #zeros and #clues ----

def count_zeros(s: str) -> int:
    s = str(s)
    return s.count("0") + s.count(".")

df["zeros"] = df[puzzle_col].apply(count_zeros)
df["clues"] = 81 - df["zeros"]

print(f"Zero Counted")

EMPTY_BINS = [
    ("tiny", 0, 4),
    ("very_easy", 5, 8),
    ("easy", 9, 16),
    ("moderate", 17, 25),
    ("medium", 26, 35),
    ("tricky", 36, 45),
    ("hard", 46, 60),
]

def assign_difficulty(n_zeros: int) -> str:
    for name, lo, hi in EMPTY_BINS:
        if lo <= n_zeros <= hi:
            return name
    return "other"

df["difficulty"] = df["zeros"].apply(assign_difficulty)

# ---- 4. Optionally subsample per difficulty ----

MAX_PER = {
    "tiny": 10_000,
    "very_easy": 20_000,
    "easy": 100_000,
    "moderate": 100_000,
    "medium": 100_000,
    "tricky": 100_000,
    "hard": 100_000,
}

dfs = []
rng = np.random.default_rng(42)

print(f"Iterating thourgh the sudoku")

for diff, max_n in MAX_PER.items():
    sub = df[df["difficulty"] == diff]
    if diff == "other":
        continue
    if len(sub) == 0:
        continue
    if len(sub) == 0:
        continue
    if len(sub) > max_n:
        # sample without replacement for reproducibility
        idx = rng.choice(sub.index.to_numpy(), size=max_n, replace=False)
        sub = sub.loc[idx]
    out_path = OUTPUT_DIR / f"sudoku_{diff}.csv"
    sub.to_csv(out_path, index=False)
    dfs.append(sub)

# Combined filtered dataset
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined_path = OUTPUT_DIR / "sudoku_all_filtered.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote combined filtered dataset to {combined_path}")
else:
    print("No puzzles matched any difficulty thresholds.")
