from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import json

import numpy as np
import math
import torch

from .distance_regressor import DistanceRegressor, IsotonicCalibrator


Board = np.ndarray  # shape (9, 9), dtype int8, values in 0..9 (0 = empty)
_LOG_TABLE = [0.0] + [math.log(i) for i in range(1, 10)]
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(512)], dtype=np.int8)
_BLOCK_IDX_GRID = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                             [0, 0, 0, 1, 1, 1, 2, 2, 2],
                             [0, 0, 0, 1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 4, 4, 4, 5, 5, 5],
                             [3, 3, 3, 4, 4, 4, 5, 5, 5],
                             [3, 3, 3, 4, 4, 4, 5, 5, 5],
                             [6, 6, 6, 7, 7, 7, 8, 8, 8],
                             [6, 6, 6, 7, 7, 7, 8, 8, 8],
                             [6, 6, 6, 7, 7, 7, 8, 8, 8]], dtype=np.int8)

# --------- Module-level caches (per process) to avoid reloading models per env ---------
_DIST_CACHE: dict[str, object] = {"model": None, "calibrator": None, "model_path": None, "calib_path": None, "device": None}


@dataclass
class StepResult:
    obs: Board
    reward: float
    done: bool
    info: Dict[str, Any]


class SudokuEnv:
    """
    Minimal RL-style Sudoku environment (no Gym/Puffer dependencies).

    - State: 9x9 board, 0 = empty, 1..9 = digits.
    - Action: integer in [0, 9*9*9), encoding (row, col, digit).
    """

    # Reward scalars (distance-driven)
    SOLVE_BONUS: float = 10.0
    WRONG_DIGIT_PENALTY: float = 3.0
    UNSOLVABLE_PENALTY: float = 20.0

    n_rows: int = 9
    n_cols: int = 9
    n_digits: int = 9

    def __init__(
        self,
        initial_board: Board,
        solution_board: Board,
        max_steps: int = 200,
        terminate_on_wrong_digit: bool = False,
        distance_model_path: Optional[Path] = None,
        calibrator_path: Optional[Path] = None,
    ) -> None:
        board, solution = initial_board, solution_board

        self.initial_board: Board = board
        self.board: Board = board.copy()
        self.solution_board: Board = solution
        self.max_steps: int = max_steps
        self.steps: int = 0
        self._recompute_masks()
        self.entropy = 0.0  # placeholder; entropy no longer used for reward
        self.initial_empties: int = int(np.sum(self.board == 0))
        self.total_reward: float = 0.0
        self.wrong_digit_count: int = 0
        self.terminate_on_wrong_digit: bool = terminate_on_wrong_digit

        # Distance model + calibrator (loaded once)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = distance_model_path or Path("experiments/distance_regressor.pt")
        calib_path = calibrator_path or Path("experiments/distance_calibrator.json")
        self.distance_model = self._load_distance_model(model_path, self.device)
        self.calibrator = self._load_calibrator(calib_path)
        self.current_F: float = self._predict_F(self.board)
        self.start_F: float = self.current_F
        self.total_delta_F: float = 0.0

        # Derived sizes
        self.n_actions: int = self.n_rows * self.n_cols * self.n_digits

    # ------------- Public API -------------

    def reset(self, seed: Optional[int] = None, initial_board: Optional[Board] = None, solution_board: Optional[Board] = None) -> Board:
        if (initial_board is None) ^ (solution_board is None):
            raise ValueError("initial_board and solution_board must be provided together")

        if initial_board is not None and solution_board is not None:
            board, solution = initial_board, solution_board
            self.initial_board = board
            self.solution_board = solution

        # seed kept for later compatibility, not used yet
        self.board = self.initial_board.copy()
        self.steps = 0
        self._recompute_masks()
        self.entropy = 0.0
        self.initial_empties = int(np.sum(self.board == 0))
        self.total_reward = 0.0
        self.wrong_digit_count = 0
        self.current_F = self._predict_F(self.board)
        self.start_F = self.current_F
        self.total_delta_F = 0.0
        return self.board.copy()

    def step(self, action: int) -> Tuple[Board, float, bool, Dict[str, Any]]:
        """
        action: int in [0, 729) encoding (row, col, digit).
        Returns: (obs, reward, done, info)
        """
        self.steps += 1
        row, col, digit = self.decode_action(action)

        reward = 0.0
        illegal_code = self._illegal_code(row, col, digit)
        illegal = illegal_code != 0
        wrong_digit = False
        solved_now = False
        delta_F = 0.0
        F_before = self.current_F
        F_after = F_before

        if illegal:
            F_after = F_before
        elif digit != self.solution_board[row, col]:
            wrong_digit = True
            reward -= self.WRONG_DIGIT_PENALTY
            self.wrong_digit_count += 1
        else:
            # Apply legal, solution-consistent move
            self.board[row, col] = digit
            self._update_masks_after_move(row, col, digit)
            F_after = self._predict_F(self.board)
            delta_F = F_before - F_after
            reward += delta_F

            if self._is_solved():
                reward += self.SOLVE_BONUS
                solved_now = True

        # Unsolvable detection: if no legal actions remain and not solved
        if not solved_now and not illegal and not wrong_digit:
            if not legal_action_mask(self.board).any():
                reward -= self.UNSOLVABLE_PENALTY

        timeout = self.steps >= self.max_steps and not solved_now

        done = solved_now or timeout or (self.terminate_on_wrong_digit and wrong_digit)
        # Track episodic accumulators
        self.total_reward += reward
        self.total_delta_F += delta_F
        self.current_F = F_after

        obs = self.board.copy()
        info = {
            "illegal": illegal,
            "solved": solved_now,
            "timeout": timeout,
            "steps": self.steps,
            "wrong_digit": wrong_digit,
            "F_before": float(F_before),
            "F_after": float(F_after),
            "delta_F": float(delta_F),
        }

        return obs, reward, done, info

    # ------------- Action encoding -------------

    def encode_action(self, row: int, col: int, digit: int) -> int:
        """
        Map (row, col, digit) to scalar action in [0, 728].
        digit is 1..9 here.
        """
        if not (0 <= row < self.n_rows):
            raise ValueError(f"row out of bounds: {row}")
        if not (0 <= col < self.n_cols):
            raise ValueError(f"col out of bounds: {col}")
        if not (1 <= digit <= self.n_digits):
            raise ValueError(f"digit must be in [1, 9], got {digit}")

        return row * (self.n_cols * self.n_digits) + col * self.n_digits + (digit - 1)

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Inverse of encode_action.
        Returns (row, col, digit), where digit in 1..9.
        """
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")

        row = action // (self.n_cols * self.n_digits)
        rem = action % (self.n_cols * self.n_digits)
        col = rem // self.n_digits
        digit_index = rem % self.n_digits
        digit = digit_index + 1
        return row, col, digit

    # ------------- Oracle / constraint checking -------------

    def _is_empty(self, row: int, col: int) -> bool:
        return self.board[row, col] == 0

    def _illegal_code(self, row: int, col: int, digit: int) -> int:
        """
        Return 0 if legal, else a small code:
        1=out_of_bounds, 2=invalid_digit, 3=overwrite, 4=row_conflict,
        5=col_conflict, 6=block_conflict.
        """
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            return 1
        if not (1 <= digit <= self.n_digits):
            return 2
        if not self._is_empty(row, col):
            return 3
        bit = 1 << (digit - 1)
        if self._row_mask[row] & bit:
            return 4
        if self._col_mask[col] & bit:
            return 5
        block_idx = (row // 3) * 3 + (col // 3)
        if self._block_mask[block_idx] & bit:
            return 6
        return 0

    def _is_solved(self) -> bool:
        """Solved = no zeros + globally valid."""
        if np.any(self.board == 0):
            return False
        # No duplicates in rows/cols/blocks if bitmasks have no overlapping bits beyond presence.
        for r in range(9):
            if self._row_mask[r].bit_count() != np.count_nonzero(self.board[r, :]):
                return False
        for c in range(9):
            if self._col_mask[c].bit_count() != np.count_nonzero(self.board[:, c]):
                return False
        for b in range(9):
            br = (b // 3) * 3
            bc = (b % 3) * 3
            block_vals = self.board[br:br+3, bc:bc+3]
            if self._block_mask[b].bit_count() != np.count_nonzero(block_vals):
                return False
        return True

    def _recompute_masks_and_entropy(self) -> None:
        self._recompute_masks()

    def _recompute_masks(self) -> None:
        self._row_mask, self._col_mask, self._block_mask, self._block_grid = _masks_from_board(self.board)
        union = (self._row_mask[:, None] | self._col_mask[None, :] | self._block_grid).astype(np.uint16)
        pop = _POPCOUNT_TABLE[union]
        counts = (9 - pop).astype(np.int8)
        empties = self.board == 0
        counts[~empties] = 0
        self.candidate_counts = counts

    def _entropy_from_counts(self, counts: np.ndarray, empties: np.ndarray, weight: float) -> float:
        if not np.any(empties):
            return 0.0
        if np.any(counts[empties] == 0):
            return float("inf")
        return float(np.log(counts[empties].astype(np.float64)).sum() + weight * empties.sum())

    def _candidate_count_fast(self, row: int, col: int) -> int:
        union = int(self._row_mask[row] | self._col_mask[col] | self._block_grid[row, col])
        return 9 - union.bit_count()

    def _update_masks_after_move(self, row: int, col: int, digit: int) -> None:
        """Incrementally update masks after placing a digit in an empty cell."""
        bit = np.uint16(1 << (digit - 1))
        self._row_mask[row] |= bit
        self._col_mask[col] |= bit
        block_idx = (row // 3) * 3 + (col // 3)
        self._block_mask[block_idx] |= bit
        block_val = self._block_mask[block_idx]
        br = (row // 3) * 3
        bc = (col // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                self._block_grid[rr, cc] = block_val


    # ---------- Distance model helpers ----------

    def _load_distance_model(self, path: Path, device) -> DistanceRegressor:
        global _DIST_CACHE
        if _DIST_CACHE["model"] is not None and _DIST_CACHE["model_path"] == path and _DIST_CACHE["device"] == str(device):
            return _DIST_CACHE["model"]
        if not path.exists():
            raise FileNotFoundError(f"Distance regressor not found at {path}")
        model = DistanceRegressor().to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        _DIST_CACHE.update(model=model, model_path=path, device=str(device))
        return model

    def _load_calibrator(self, path: Path) -> IsotonicCalibrator:
        global _DIST_CACHE
        if _DIST_CACHE["calibrator"] is not None and _DIST_CACHE["calib_path"] == path:
            return _DIST_CACHE["calibrator"]
        if not path.exists():
            raise FileNotFoundError(f"Calibrator json not found at {path}")
        data = json.loads(path.read_text())
        calib = IsotonicCalibrator.from_dict(data)
        _DIST_CACHE.update(calibrator=calib, calib_path=path)
        return calib

    def _predict_distance(self, board: Board) -> float:
        x = torch.as_tensor(board.reshape(1, -1), dtype=torch.float32, device=self.device) / 9.0
        with torch.no_grad():
            pred = self.distance_model(x).item()
        return float(pred)

    def _predict_F(self, board: Board) -> float:
        d = np.array([self._predict_distance(board)], dtype=np.float64)
        return float(self.calibrator.predict(d)[0])


def legal_action_mask(board: Board) -> np.ndarray:
    """Return a boolean mask of legal (row, col, digit) actions for `board`."""
    if board.shape != (9, 9):
        raise ValueError(f"board must be 9x9, got {board.shape}")

    mask = np.zeros(9 * 9 * 9, dtype=bool)

    for row in range(9):
        for col in range(9):
            if board[row, col] != 0:
                continue

            row_vals = board[row, :]
            col_vals = board[:, col]
            block_row = (row // 3) * 3
            block_col = (col // 3) * 3
            block_vals = board[block_row:block_row + 3, block_col:block_col + 3]

            for digit in range(1, 10):
                if digit in row_vals:
                    continue
                if digit in col_vals:
                    continue
                if digit in block_vals:
                    continue

                action = row * (9 * 9) + col * 9 + (digit - 1)
                mask[action] = True

    return mask

def count_violations(board: Board) -> int:
    """
    Count total Sudoku constraint violations:
    - duplicates in each row
    - duplicates in each column
    - duplicates in each 3x3 block
    """
    v = 0

    # Rows
    for r in range(9):
        row = board[r, :]
        non_zero = row[row != 0]
        _, counts = np.unique(non_zero, return_counts=True)
        v += np.sum(counts - 1)

    # Cols
    for c in range(9):
        col = board[:, c]
        non_zero = col[col != 0]
        _, counts = np.unique(non_zero, return_counts=True)
        v += np.sum(counts - 1)

    # Blocks
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = board[br:br+3, bc:bc+3].reshape(-1)
            non_zero = block[block != 0]
            _, counts = np.unique(non_zero, return_counts=True)
            v += np.sum(counts - 1)

    return int(v)


def count_empties(board: Board) -> int:
    """Return number of empty cells (zeros)."""
    return int(np.sum(board == 0))


def candidate_count(
    board: Board,
    row: int,
    col: int,
    masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> int:
    """
    Number of digits 1..9 that can legally be placed at (row, col)
    given the current board constraints. Returns 0 for already-filled cells.
    """
    if board[row, col] != 0:
        return 0

    if masks is None:
        row_mask, col_mask, _, block_grid = _masks_from_board(board)
    else:
        row_mask, col_mask, block_grid = masks
    mask = row_mask[row] | col_mask[col] | block_grid[row, col]
    return 9 - int(mask.bit_count())


def board_entropy(
    board: Board,
    entropy_empty_weight: float = 0.0,
    masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    """
    Sum of log candidate counts across empty cells.
    Lower is better; solved boards have entropy 0.
    """
    if masks is None:
        row_mask, col_mask, block_mask, block_grid = _masks_from_board(board)
    else:
        row_mask, col_mask, block_grid = masks
    empties = board == 0
    if not np.any(empties):
        return 0.0

    union = (row_mask[:, None] | col_mask[None, :] | block_grid).astype(np.uint16)
    union = np.bitwise_and(union, 0x1FF)[empties]
    popcounts = np.frompyfunc(int.bit_count, 1, 1)(union.astype(int)).astype(np.int8)
    counts = 9 - popcounts
    if np.any(counts == 0):
        return float("inf")
    entropy = float(np.log(counts.astype(np.float64)).sum()) + entropy_empty_weight * float(len(counts))
    return entropy


def _masks_from_board(board: Board):
    board = np.asarray(board, dtype=np.int8)
    if board.shape != (9, 9):
        raise ValueError(f"board must be 9x9, got {board.shape}")

    row_mask = np.zeros(9, dtype=np.uint16)
    col_mask = np.zeros(9, dtype=np.uint16)
    block_mask = np.zeros(9, dtype=np.uint16)

    nonzero = board != 0
    if np.any(nonzero):
        rows, cols = np.nonzero(nonzero)
        vals = board[rows, cols]
        bits = np.left_shift(np.uint16(1), vals.astype(np.int32) - 1)
        np.bitwise_or.at(row_mask, rows, bits)
        np.bitwise_or.at(col_mask, cols, bits)
        block_indices = (rows // 3) * 3 + (cols // 3)
        np.bitwise_or.at(block_mask, block_indices, bits)

    block_grid = block_mask[_BLOCK_IDX_GRID]

    return row_mask, col_mask, block_mask, block_grid
