import numpy as np

from sudoku_rl.env import candidate_count, board_entropy, _masks_from_board


def test_candidate_count_empty_board_is_nine():
    board = np.zeros((9, 9), dtype=np.int8)
    assert candidate_count(board, 0, 0) == 9
    assert candidate_count(board, 8, 8) == 9


def test_candidate_count_respects_row_col_and_block():
    board = np.zeros((9, 9), dtype=np.int8)
    board[0, 0] = 5  # same row
    board[1, 2] = 6  # same column
    board[1, 1] = 7  # same 3x3 block
    # At (0,2) candidates should exclude 5 (row), 6 (col), 7 (block)
    assert candidate_count(board, 0, 2) == 6  # 9 - {5,6,7}


def test_board_entropy_empty_board_matches_closed_form():
    board = np.zeros((9, 9), dtype=np.int8)
    expected = 81 * np.log(9)
    assert np.isclose(board_entropy(board), expected)


def test_board_entropy_becomes_infinite_when_no_candidates():
    board = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 9],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    # The empty cell at (0,8) sees digits 1..9 already used in row/col, so no candidates.
    assert board_entropy(board) == float("inf")


def test_board_entropy_zero_when_only_singleton_candidates_left():
    board = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 0],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.int8,
    )
    # Only (0,8) is empty and its only legal digit is 9, so log(1) = 0.
    assert np.isclose(board_entropy(board), 0.0)


def test_board_entropy_cached_matches_uncached():
    board = np.array(
        [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ],
        dtype=np.int8,
    )

    masks = _masks_from_board(board)
    direct = board_entropy(board, entropy_empty_weight=0.5)
    cached = board_entropy(board, entropy_empty_weight=0.5, masks=(masks[0], masks[1], masks[3]))
    assert np.isclose(direct, cached)
