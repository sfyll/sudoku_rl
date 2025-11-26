import numpy as np

from sudoku_rl.env import board_entropy, candidate_count


def _reference_candidate_count(board, row, col):
    if board[row, col] != 0:
        return 0
    block_row = (row // 3) * 3
    block_col = (col // 3) * 3
    block = board[block_row:block_row + 3, block_col:block_col + 3]
    used = set(board[row, :]) | set(board[:, col]) | set(block.reshape(-1))
    used.discard(0)
    return 9 - len(used)


def _reference_entropy(board, entropy_empty_weight=0.0):
    ent = 0.0
    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                continue
            cnt = _reference_candidate_count(board, r, c)
            if cnt == 0:
                return float("inf")
            ent += np.log(cnt) + entropy_empty_weight
    return float(ent)


def _random_board(seed=0):
    rng = np.random.default_rng(seed)
    board = np.zeros((9, 9), dtype=np.int8)
    cells = rng.choice(81, size=25, replace=False)
    for idx in cells:
        r, c = divmod(idx, 9)
        board[r, c] = int(rng.integers(1, 10))
    return board


def test_candidate_count_matches_reference():
    for seed in range(5):
        board = _random_board(seed)
        for r in range(9):
            for c in range(9):
                assert candidate_count(board, r, c) == _reference_candidate_count(board, r, c)


def test_entropy_matches_reference():
    for seed in range(5):
        board = _random_board(seed)
        ref = _reference_entropy(board, entropy_empty_weight=0.5)
        fast = board_entropy(board, entropy_empty_weight=0.5)
        if np.isinf(ref):
            assert np.isinf(fast)
        else:
            assert abs(ref - fast) < 1e-6
