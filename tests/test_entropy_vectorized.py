import numpy as np
import pytest

from sudoku_rl.env import board_entropy
from sudoku_rl.env import SudokuEnv


def _reference_entropy(board, weight=0.5):
    ent = 0.0
    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                continue
            row_vals = set(board[r, :])
            col_vals = set(board[:, c])
            br = (r // 3) * 3
            bc = (c // 3) * 3
            block = set(board[br:br+3, bc:bc+3].reshape(-1))
            used = row_vals | col_vals | block
            used.discard(0)
            cnt = 9 - len(used)
            if cnt == 0:
                return float("inf")
            ent += np.log(cnt) + weight
    return float(ent)


def random_board(seed=0):
    rng = np.random.default_rng(seed)
    board = np.zeros((9, 9), dtype=np.int8)
    cells = rng.choice(81, size=30, replace=False)
    for idx in cells:
        r, c = divmod(idx, 9)
        board[r, c] = int(rng.integers(1, 10))
    return board


def test_entropy_matches_reference_many():
    for seed in range(10):
        board = random_board(seed)
        ref = _reference_entropy(board, weight=SudokuEnv.ENTROPY_EMPTY_WEIGHT)
        fast = board_entropy(board, entropy_empty_weight=SudokuEnv.ENTROPY_EMPTY_WEIGHT)
        if np.isinf(ref):
            assert np.isinf(fast)
        else:
            assert abs(ref - fast) < 1e-6
