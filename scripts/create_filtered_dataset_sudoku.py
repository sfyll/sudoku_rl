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

def assign_difficulty(n_zeros: int) -> str:
    if n_zeros <= 4:
        return "super_easy"
    elif n_zeros <= 25:
        return "easy"
    elif n_zeros <= 45:
        return "medium"
    elif n_zeros <= 60:
        return "hard"

df["difficulty"] = df["zeros"].apply(assign_difficulty)

# ---- 4. Optionally subsample per difficulty ----

MAX_PER = {
    "super_easy": 5_000,
    "easy": 50_000,
    "medium": 50_000,
    "hard": 50_000,
}

dfs = []
rng = np.random.default_rng(42)

print(f"Iterating thourgh the sudoku")

for diff, max_n in MAX_PER.items():
    sub = df[df["difficulty"] == diff]
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

