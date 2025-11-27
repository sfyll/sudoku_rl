from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np

try:
    import numba as nb
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False


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

    # Reward scalars (entropy-driven)
    SOLVE_BONUS: float = 10.0
    WRONG_DIGIT_PENALTY: float = 3.0
    # If an action would push the board into a state with zero candidates,
    # we treat that as high-entropy (unsolvable) rather than crashing.
    UNSOLVABLE_PENALTY: float = 20.0
    ENTROPY_EMPTY_WEIGHT: float = 0.5

    n_rows: int = 9
    n_cols: int = 9
    n_digits: int = 9

    def __init__(
        self,
        initial_board: Board,
        solution_board: Board,
        max_steps: int = 200,
        terminate_on_wrong_digit: bool = True,
    ) -> None:
        board, solution = self._validate_boards(initial_board, solution_board)

        self.initial_board: Board = board
        self.board: Board = board.copy()
        self.solution_board: Board = solution
        self.max_steps: int = max_steps
        self.steps: int = 0
        self.initial_empties: int = int(np.sum(self.board == 0))
        self.start_entropy: float = board_entropy(self.board, self.ENTROPY_EMPTY_WEIGHT)
        self.total_reward: float = 0.0
        self.total_entropy_delta: float = 0.0
        self.wrong_digit_count: int = 0
        self.terminate_on_wrong_digit: bool = terminate_on_wrong_digit

        # Derived sizes
        self.n_actions: int = self.n_rows * self.n_cols * self.n_digits

    # ------------- Public API -------------

    def reset(self, seed: Optional[int] = None, initial_board: Optional[Board] = None, solution_board: Optional[Board] = None) -> Board:
        if (initial_board is None) ^ (solution_board is None):
            raise ValueError("initial_board and solution_board must be provided together")

        if initial_board is not None and solution_board is not None:
            board, solution = self._validate_boards(initial_board, solution_board)
            self.initial_board = board
            self.solution_board = solution

        # seed kept for later compatibility, not used yet
        self.board = self.initial_board.copy()
        self.steps = 0
        self.initial_empties = int(np.sum(self.board == 0))
        self.start_entropy = board_entropy(self.board, self.ENTROPY_EMPTY_WEIGHT)
        self.total_reward = 0.0
        self.total_entropy_delta = 0.0
        self.wrong_digit_count = 0
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
        entropy_delta = 0.0

        entropy_before = board_entropy(self.board, self.ENTROPY_EMPTY_WEIGHT)

        if illegal:
            entropy_after = entropy_before
        elif digit != self.solution_board[row, col]:
            wrong_digit = True
            entropy_after = entropy_before
            entropy_delta = 0.0
            reward -= self.WRONG_DIGIT_PENALTY
            self.wrong_digit_count += 1
        else:
            # Apply legal, solution-consistent move
            self.board[row, col] = digit
            entropy_after = board_entropy(self.board, self.ENTROPY_EMPTY_WEIGHT)
            if np.isfinite(entropy_after):
                entropy_delta = entropy_before - entropy_after
            else:
                # Something made the puzzle unsatisfiable; treat as strong penalty.
                print("Unsolvable!")
                entropy_delta = -self.UNSOLVABLE_PENALTY
            reward += entropy_delta

            if self._is_solved():
                reward += self.SOLVE_BONUS
                solved_now = True

        timeout = self.steps >= self.max_steps and not solved_now

        done = solved_now or timeout or (self.terminate_on_wrong_digit and wrong_digit)
        # Track episodic accumulators
        self.total_reward += reward
        self.total_entropy_delta += entropy_delta

        obs = self.board.copy()
        info = {
            "illegal": illegal,
            "solved": solved_now,
            "timeout": timeout,
            "steps": self.steps,
            "wrong_digit": wrong_digit,
            "entropy_before": float(entropy_before),
            "entropy_after": float(entropy_after),
            "entropy_delta": float(entropy_delta),
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

    def _validate_boards(self, initial_board: Board, solution_board: Board) -> Tuple[Board, Board]:
        """Validate shapes/values and ensure solution is fully specified."""
        board = np.array(initial_board, dtype=np.int8, copy=True)
        if board.shape != (self.n_rows, self.n_cols):
            raise ValueError(f"initial_board must be 9x9, got {board.shape}")
        if np.any((board < 0) | (board > 9)):
            raise ValueError("initial_board entries must be in [0, 9]")

        solution = np.array(solution_board, dtype=np.int8, copy=True)
        if solution.shape != (self.n_rows, self.n_cols):
            raise ValueError(f"solution_board must be 9x9, got {solution.shape}")
        if np.any((solution < 1) | (solution > 9)):
            raise ValueError("solution_board entries must be in [1, 9]")
        if np.any(solution == 0):
            raise ValueError("solution_board must be fully solved (no zeros).")

        fixed_mask = board != 0
        if np.any(board[fixed_mask] != solution[fixed_mask]):
            raise ValueError("initial_board conflicts with solution_board at fixed cells.")

        return board, solution


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


def candidate_count(board: Board, row: int, col: int) -> int:
    """
    Number of digits 1..9 that can legally be placed at (row, col)
    given the current board constraints. Returns 0 for already-filled cells.
    """
    if board[row, col] != 0:
        return 0

    if USE_NUMBA:
        return _candidate_count_nb(board, row, col)

    row_mask, col_mask, block_mask, block_grid = _masks_from_board(board)
    mask = row_mask[row] | col_mask[col] | block_grid[row, col]
    return 9 - int(mask.bit_count())


def board_entropy(board: Board, entropy_empty_weight: float = 0.0) -> float:
    """
    Sum of log candidate counts across empty cells.
    Lower is better; solved boards have entropy 0.
    """
    if USE_NUMBA:
        return float(_board_entropy_nb(board, entropy_empty_weight))

    row_mask, col_mask, block_mask, block_grid = _masks_from_board(board)
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
        for r, c, b in zip(rows, cols, bits):
            row_mask[r] |= b
            col_mask[c] |= b
            block_idx = (r // 3) * 3 + (c // 3)
            block_mask[block_idx] |= b

    block_grid = np.empty((9, 9), dtype=np.uint16)
    for r in range(9):
        for c in range(9):
            block_grid[r, c] = block_mask[(r // 3) * 3 + (c // 3)]

    return row_mask, col_mask, block_mask, block_grid


# ---------------- Numba-accelerated helpers ----------------
if USE_NUMBA:

    @nb.njit(cache=True)
    def _masks_from_board_nb(board: np.ndarray):
        row_mask = np.zeros(9, dtype=np.uint16)
        col_mask = np.zeros(9, dtype=np.uint16)
        block_mask = np.zeros(9, dtype=np.uint16)

        for r in range(9):
            for c in range(9):
                v = board[r, c]
                if v == 0:
                    continue
                b = np.uint16(1 << (v - 1))
                row_mask[r] |= b
                col_mask[c] |= b
                block_idx = (r // 3) * 3 + (c // 3)
                block_mask[block_idx] |= b

        block_grid = np.empty((9, 9), dtype=np.uint16)
        for r in range(9):
            for c in range(9):
                block_grid[r, c] = block_mask[(r // 3) * 3 + (c // 3)]

        return row_mask, col_mask, block_mask, block_grid


    @nb.njit(cache=True)
    def _candidate_count_nb(board: np.ndarray, row: int, col: int) -> int:
        if board[row, col] != 0:
            return 0
        row_mask, col_mask, block_mask, block_grid = _masks_from_board_nb(board)
        mask = row_mask[row] | col_mask[col] | block_grid[row, col]
        # popcount for 9 bits
        count = 0
        tmp = mask
        while tmp:
            tmp &= tmp - 1
            count += 1
        return 9 - count


    @nb.njit(cache=True)
    def _board_entropy_nb(board: np.ndarray, entropy_empty_weight: float) -> float:
        row_mask, col_mask, block_mask, block_grid = _masks_from_board_nb(board)
        entropy = 0.0
        empty_found = False
        for r in range(9):
            for c in range(9):
                if board[r, c] != 0:
                    continue
                empty_found = True
                mask = row_mask[r] | col_mask[c] | block_grid[r, c]
                count = 9
                tmp = mask
                while tmp:
                    tmp &= tmp - 1
                    count -= 1
                if count == 0:
                    return float("inf")
                entropy += np.log(float(count)) + entropy_empty_weight
        if not empty_found:
            return 0.0
        return entropy
