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

        # Derived sizes
        self.n_actions: int = self.n_rows * self.n_cols * self.n_digits

    # ------------- Public API -------------

    def reset(self, seed: Optional[int] = None) -> Board:
        # seed kept for later compatibility, not used yet
        self.board = self.initial_board.copy()
        self.steps = 0
        return self.board.copy()

    def step(self, action: int) -> Tuple[Board, float, bool, Dict[str, Any]]:
        """
        action: int in [0, 729) encoding (row, col, digit).
        Returns: (obs, reward, done, info)
        """
        self.steps += 1

        # Decode (row, col, digit)
        row, col, digit = self.decode_action(action)

        reward = -0.01  # small step penalty
        illegal = False
        solved_now = False

        # Illegal move: wrong position, overwriting, or breaking constraints.
        if not self._is_valid_move(row, col, digit):
            illegal = True
            reward -= 1.0  # strong negative signal
            # Board unchanged.
        else:
            # Apply move (we know cell was empty and move is legal)
            self.board[row, col] = digit

           # Simple shaping: reward filling a previously empty cell
            reward += 0.1
            if self._is_solved():
                reward += 1.0
                solved_now = True
        timeout = self.steps >= self.max_steps and not solved_now
        if timeout:
            # Optional: extra failure penalty
            reward -= 0.5

        done = solved_now or timeout
        obs = self.board.copy()
        info = {
            "illegal": illegal,
            "solved": solved_now,
            "timeout": timeout,
            "steps": self.steps,
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

    def _is_valid_move(self, row: int, col: int, digit: int) -> bool:
        """Check if placing `digit` at (row, col) is legal on current board."""
        # Bounds
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            return False
        if not (1 <= digit <= self.n_digits):
            return False

        # Cell must be empty: no overwriting.
        if self.board[row, col] != 0:
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

