import json
from pathlib import Path

import numpy as np
import pandas as pd


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

# ---- 3. Auto-create 4-hole bins ----

BIN_SIZE = 4
zeros_min = int(df["zeros"].min())
zeros_max = int(df["zeros"].max())

def make_bin_edges(lo: int, hi: int, step: int):
    edges = []
    start = 0  # always start at 0 blanks for consistent bin labels
    for b in range(start, hi + step, step):
        edges.append((b, b + step - 1))
    return edges

BIN_RANGES = make_bin_edges(zeros_min, zeros_max, BIN_SIZE)

def bin_label(lo: int, hi: int) -> str:
    return f"zeros_{lo:02d}_{hi:02d}"

df["bin_label"] = pd.cut(
    df["zeros"],
    bins=[r[0] - 0.5 for r in BIN_RANGES] + [BIN_RANGES[-1][1] + 0.5],
    labels=[bin_label(lo, hi) for lo, hi in BIN_RANGES],
    include_lowest=True,
)

# ---- 4. Subsample per bin (uniform cap) ----

MAX_PER_BIN = 100_000
rng = np.random.default_rng(42)
dfs = []
manifest = []

print("Iterating through bins")

for (lo, hi) in BIN_RANGES:
    label = bin_label(lo, hi)
    sub = df[df["bin_label"] == label]
    out_path = OUTPUT_DIR / f"sudoku_{label}.csv"

    if len(sub) > 0:
        if len(sub) > MAX_PER_BIN:
            idx = rng.choice(sub.index.to_numpy(), size=MAX_PER_BIN, replace=False)
            sub = sub.loc[idx]
        sub_to_write = sub[[puzzle_col, solution_col, "zeros", "clues", "bin_label"]]
        sub_to_write.to_csv(out_path, index=False)
        dfs.append(sub_to_write)
        rows_written = int(len(sub_to_write))
    else:
        # Write an empty placeholder with headers so the manifest is complete.
        sub_to_write = pd.DataFrame(columns=[puzzle_col, solution_col, "zeros", "clues", "bin_label"])
        sub_to_write.to_csv(out_path, index=False)
        rows_written = 0

    manifest.append(
        {
            "label": label,
            "path": out_path.name,
            "rows": rows_written,
            "zeros_range": [lo, hi],
        }
    )

# Combined filtered dataset
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined_path = OUTPUT_DIR / "sudoku_all_filtered.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote combined filtered dataset to {combined_path}")

    manifest_path = OUTPUT_DIR / "sudoku_bins_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote bin manifest to {manifest_path}")
else:
    print("No puzzles matched any bin thresholds.")
