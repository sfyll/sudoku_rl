from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np


Board = np.ndarray  # shape (9, 9), dtype int8, values in 0..9 (0 = empty)


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

    n_rows: int = 9
    n_cols: int = 9
    n_digits: int = 9

    def __init__(
        self,
        initial_board: Optional[Board] = None,
        max_steps: int = 200,
        solution_board: Optional[Board] = None,
    ) -> None:
        if initial_board is None:
            board = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        else:
            board = np.array(initial_board, dtype=np.int8, copy=True)
            if board.shape != (self.n_rows, self.n_cols):
                raise ValueError(f"initial_board must be 9x9, got {board.shape}")
            if np.any((board < 0) | (board > 9)):
                raise ValueError("initial_board entries must be in [0, 9]")

        self.initial_board: Board = board
        self.board: Board = board.copy()
        self.max_steps: int = max_steps
        self.steps: int = 0
        self.initial_empties: int = int(np.sum(self.board == 0))
        self.solution_board: Optional[Board] = None
        if solution_board is not None:
            if solution_board.shape != (self.n_rows, self.n_cols):
                raise ValueError(f"solution_board must be 9x9, got {solution_board.shape}")
            self.solution_board = solution_board.copy()

        # Derived sizes
        self.n_actions: int = self.n_rows * self.n_cols * self.n_digits

    # ------------- Public API -------------

    def reset(self, seed: Optional[int] = None, initial_board: Optional[Board] = None, solution_board: Optional[Board] = None) -> Board:
        if initial_board is not None:
            if initial_board.shape != (self.n_rows, self.n_cols):
                raise ValueError(f"initial_board must be 9x9, got {initial_board.shape}")
            self.initial_board = initial_board.copy()
            self.board = self.initial_board.copy()
            if solution_board is not None:
                if solution_board.shape != (self.n_rows, self.n_cols):
                    raise ValueError(f"solution_board must be 9x9, got {solution_board.shape}")
                self.solution_board = solution_board.copy()
        else:
            # seed kept for later compatibility, not used yet
            self.board = self.initial_board.copy()
        self.steps = 0
        self.initial_empties = int(np.sum(self.board == 0))
        return self.board.copy()

    def step(self, action: int) -> Tuple[Board, float, bool, Dict[str, Any]]:
        """
        action: int in [0, 729) encoding (row, col, digit).
        Returns: (obs, reward, done, info)
        """
        # --- Reward shaping constants (tune these) ---
        STEP_PENALTY     = -0.01
        ILLEGAL_PENALTY  = -1.0
        MISTAKE_PENALTY  = -0.025
        FILL_BONUS       = 0.10   # reward for safe progress: easier bump
        SOLVE_BONUS      = 3.0
        EMPTY_WEIGHT     = 0.02   # reward per reduction in empty cells
        VIOLATION_WEIGHT = 0.02   # reward per reduction in violations
        TIMEOUT_PENALTY  = -0.5
        NO_LEGAL_PENALTY = -3.0   # strong signal: dead-end is costly
        # Rough guide to magnitudes (before advantage normalization):
        # - legal fill, still unsolved (empties -1, no new violations): -0.01 + 0.05 + 0.02 ≈ +0.06
        # - final solving move: previous + SOLVE_BONUS → ≈ +3.06
        # - illegal overwrite (codes 1–3): -0.01 -1.0 ≈ -1.01
        # - wrong-digit conflict (codes 4–6): -0.01 -0.05 ≈ -0.06
        # - timeout adds -0.5 on the last step if not solved.


        self.steps += 1
        row, col, digit = self.decode_action(action)


        reward = STEP_PENALTY
        illegal = False
        solved_now = False

        # Pre-move stats for dense shaping
        empties_before = count_empties(self.board)
        viol_before = count_violations(self.board)

        illegal_code = self._illegal_code(row, col, digit)
        if illegal_code:
            # Board unchanged, strong negative signal
            illegal = True
            if illegal_code <= 3:
                reward += ILLEGAL_PENALTY
            else:
                reward += MISTAKE_PENALTY
        elif self.solution_board is not None and digit != self.solution_board[row, col]:
            # Move is locally legal but does not match the ground-truth solution.
            illegal = True
            illegal_code = 7  # custom code for wrong-solution digit
            reward += MISTAKE_PENALTY * 2
        else:
            # Apply legal move
            self.board[row, col] = digit

            # Small bonus for filling an empty cell
            reward += FILL_BONUS

            # Progress on empties
            empties_after = count_empties(self.board)
            reward += EMPTY_WEIGHT * (empties_before - empties_after)

            # Progress on global constraints (will be 0 as long as all moves keep board valid,
            # but becomes useful if you later relax _is_valid_move)
            viol_after = count_violations(self.board)
            reward += VIOLATION_WEIGHT * (viol_before - viol_after)

            # Terminal bonus
            if self._is_solved():
                reward += SOLVE_BONUS
                solved_now = True

        timeout = self.steps >= self.max_steps and not solved_now
        if timeout:
            reward += TIMEOUT_PENALTY

        done = solved_now or timeout
        obs = self.board.copy()
        info = {
            "illegal": illegal,
            "illegal_conflict": 1.0 if illegal_code in (4, 5, 6) else 0.0,
            "illegal_overwrite": 1.0 if illegal_code == 3 else 0.0,
            "illegal_code": illegal_code,
            "solved": solved_now,
            "timeout": timeout,
            "no_legal_moves": False,
            "steps": self.steps,
            "steps_per_empty": self.steps / max(1, self.initial_empties),
        }
        for code in range(1, 8):
            info[f"illegal_code_{code}"] = 1.0 if illegal_code == code else 0.0
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

    def _is_valid_move(self, row: int, col: int, digit: int) -> bool:
        """Check if placing `digit` at (row, col) is legal on current board."""
        # Bounds
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            return False
        if not (1 <= digit <= self.n_digits):
            return False

        # Cell must be empty: no overwriting.
        if not self._is_empty(row, col):
            return False

        # Row constraint
        if digit in self.board[row, :]:
            return False

        # Column constraint
        if digit in self.board[:, col]:
            return False

        # 3x3 block constraint
        block_row = (row // 3) * 3
        block_col = (col // 3) * 3
        block = self.board[block_row:block_row + 3, block_col:block_col + 3]
        if digit in block:
            return False

        return True

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
        if digit in self.board[row, :]:
            return 4
        if digit in self.board[:, col]:
            return 5
        block_row = (row // 3) * 3
        block_col = (col // 3) * 3
        block = self.board[block_row:block_row + 3, block_col:block_col + 3]
        if digit in block:
            return 6
        return 0

    def _groups_valid(self, values: np.ndarray) -> bool:
        """
        Helper: check that no digit 1..9 appears more than once
        in a 1D vector (ignore zeros).
        """
        non_zero = values[values != 0]
        if non_zero.size == 0:
            return True
        # Since domain is small, simple check is fine.
        uniq, counts = np.unique(non_zero, return_counts=True)
        return np.all(counts <= 1)

    def _board_valid(self) -> bool:
        """Check entire board is Sudoku-valid (no conflicts), ignoring empties."""
        # Rows
        for r in range(self.n_rows):
            if not self._groups_valid(self.board[r, :]):
                return False

        # Columns
        for c in range(self.n_cols):
            if not self._groups_valid(self.board[:, c]):
                return False

        # 3x3 blocks
        for br in range(0, self.n_rows, 3):
            for bc in range(0, self.n_cols, 3):
                block = self.board[br:br + 3, bc:bc + 3].reshape(-1)
                if not self._groups_valid(block):
                    return False

        return True

    def _is_solved(self) -> bool:
        """Solved = no zeros + globally valid."""
        if np.any(self.board == 0):
            return False
        return self._board_valid()


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
