from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .solver_norvig import solve_with_trace
from .puzzle import sample_puzzle, get_puzzle_pool


@dataclass
class DistanceExample:
    board: np.ndarray  # shape (81,)
    distance: float
    bin_label: str
    puzzle_id: int


class DistanceDataset(Dataset):
    def __init__(self, examples: Sequence[DistanceExample]):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        board = torch.as_tensor(ex.board, dtype=torch.float32) / 9.0
        target = torch.tensor(ex.distance, dtype=torch.float32)
        return board, target


class DistanceRegressor(nn.Module):
    def __init__(self, hidden_sizes: Tuple[int, int, int] = (512, 256, 128)):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(81, h1),
            nn.ReLU(),
            nn.LayerNorm(h1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------- Data generation ----------------

def generate_distance_examples(
    bin_labels: Sequence[str],
    puzzles_per_bin: Optional[int | dict] = 1000,
    seed_offset: int = 0,
) -> List[DistanceExample]:
    """
    Sample unique puzzles per bin (up to pool size) and build distance-labeled states.

    puzzles_per_bin:
        - int/None: same cap for every bin (<=0 means use all puzzles available)
        - dict: mapping bin_label -> cap (<=0 means all for that bin)
    """
    examples: List[DistanceExample] = []
    for bin_label in bin_labels:
        if isinstance(puzzles_per_bin, dict):
            target = puzzles_per_bin.get(bin_label, puzzles_per_bin.get("default", 0))
        else:
            target = puzzles_per_bin

        pool = get_puzzle_pool(bin_label)
        if target is None or target <= 0:
            num = len(pool)
        else:
            num = min(len(pool), int(target))

        for i in range(num):
            seed = seed_offset + i  # deterministic, unique within the chosen subset
            puzzle, solution = sample_puzzle(bin_label=bin_label, seed=seed, return_solution=True, prev_mix_ratio=0.0)
            trace = solve_with_trace(puzzle)
            if trace is None:
                continue
            T = len(trace) - 1
            for t, board in enumerate(trace):
                dist = float(T - t)
                examples.append(
                    DistanceExample(
                        board=board.reshape(-1).astype(np.float32),
                        distance=dist,
                        bin_label=bin_label,
                        puzzle_id=seed,
                    )
                )
    return examples


# ---------------- Training ----------------

def train_regressor(
    train_ds: DistanceDataset,
    val_ds: DistanceDataset,
    *,
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[DistanceRegressor, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DistanceRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.set_grad_enabled(train):
                preds = model(xb)
                loss = loss_fn(preds, yb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
        return total_loss / max(1, total_n)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        tl = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if verbose:
            print(f"[epoch {ep+1}/{epochs}] train_loss={tl:.4f} val_loss={vl:.4f}")

    return model, history


# ---------------- Evaluation helpers ----------------

def spearmanr(preds: np.ndarray, targets: np.ndarray) -> float:
    # simple rank correlation without scipy
    def rankdata(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    rx = rankdata(preds)
    ry = rankdata(targets)
    rxm = rx.mean()
    rym = ry.mean()
    cov = np.mean((rx - rxm) * (ry - rym))
    std = np.std(rx) * np.std(ry)
    return float(cov / (std + 1e-9))


def evaluate(model: DistanceRegressor, ds: DistanceDataset, device: Optional[str] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_hat = model(xb).cpu().numpy()
            preds.append(y_hat)
            targets.append(yb.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "mae": float(np.mean(np.abs(preds - targets))),
        "rmse": float(math.sqrt(np.mean((preds - targets) ** 2))),
        "spearman": spearmanr(preds, targets),
    }


# ---------------- Monotone calibration (isotonic regression) ----------------

class IsotonicCalibrator:
    """
    Lightweight 1-D monotone (non-decreasing) calibrator using Pool Adjacent Violators.
    Produces piecewise-linear interpolation over block-averaged knots.
    """

    def __init__(self):
        self.knots_x: Optional[np.ndarray] = None
        self.knots_y: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        blocks_x = x.tolist()
        blocks_y = y.tolist()
        blocks_w = [1.0] * len(x)

        i = 0
        while i < len(blocks_y) - 1:
            if blocks_y[i] <= blocks_y[i + 1] + 1e-12:
                i += 1
                continue
            # merge i and i+1
            w = blocks_w[i] + blocks_w[i + 1]
            y_new = (blocks_w[i] * blocks_y[i] + blocks_w[i + 1] * blocks_y[i + 1]) / w
            x_new = (blocks_w[i] * blocks_x[i] + blocks_w[i + 1] * blocks_x[i + 1]) / w
            blocks_w[i] = w
            blocks_y[i] = y_new
            blocks_x[i] = x_new
            del blocks_w[i + 1]
            del blocks_y[i + 1]
            del blocks_x[i + 1]
            # step back if possible to ensure previous monotonicity
            if i > 0:
                i -= 1

        self.knots_x = np.array(blocks_x, dtype=np.float64)
        self.knots_y = np.array(blocks_y, dtype=np.float64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.knots_x is None or self.knots_y is None:
            raise RuntimeError("Calibrator not fitted")
        x = np.asarray(x, dtype=np.float64)
        # clamp then interpolate
        return np.interp(x, self.knots_x, self.knots_y, left=self.knots_y[0], right=self.knots_y[-1])

    def to_dict(self) -> dict:
        return {"knots_x": self.knots_x.tolist(), "knots_y": self.knots_y.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        obj = cls()
        obj.knots_x = np.array(d["knots_x"], dtype=np.float64)
        obj.knots_y = np.array(d["knots_y"], dtype=np.float64)
        return obj



__all__ = [
    "DistanceExample",
    "DistanceDataset",
    "DistanceRegressor",
    "generate_distance_examples",
    "train_regressor",
    "evaluate",
    "IsotonicCalibrator",
    "spearmanr",
]
