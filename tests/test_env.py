import numpy as np

from sudoku_rl import SudokuEnv


def make_simple_puzzle():
    """
    Super simple, mostly empty puzzle to reason about in tests.
    5 3 . | . . . | . . .
    6 . . | . . . | . . .
    . 9 8 | . . . | . . .
    """
    board = np.zeros((9, 9), dtype=np.int8)
    board[0, 0] = 5
    board[0, 1] = 3
    board[1, 0] = 6
    board[2, 1] = 9
    board[2, 2] = 8
    return board


def test_reset_returns_initial_board_copy():
    puzzle = make_simple_puzzle()
    env = SudokuEnv(initial_board=puzzle)
    obs = env.reset()

    # Same values
    assert np.array_equal(obs, puzzle)

    # But not the same object
    obs[0, 0] = 0
    assert env.board[0, 0] == 5


def test_illegal_move_penalized_and_not_applied():
    puzzle = make_simple_puzzle()
    env = SudokuEnv(initial_board=puzzle)
    env.reset()

    # Try to place a 5 in same row, which is illegal.
    action = env.encode_action(row=0, col=2, digit=5)
    obs, reward, done, info = env.step(action)

    # Board must be unchanged at that location.
    assert obs[0, 2] == 0
    assert info["illegal"] is True
    assert reward < -0.5  # step penalty + illegal penalty
    assert done is False  # we don't terminate on illegal moves


def test_valid_move_fills_cell_and_gives_shaping_reward():
    puzzle = make_simple_puzzle()
    env = SudokuEnv(initial_board=puzzle)
    obs0 = env.reset()

    before_filled = int(np.count_nonzero(obs0))

    # Legal move: place '1' somewhere empty where it doesn't conflict.
    # Here, (0, 2) with digit 1 is fine.
    action = env.encode_action(row=0, col=2, digit=1)
    obs, reward, done, info = env.step(action)

    after_filled = int(np.count_nonzero(obs))

    assert obs[0, 2] == 1
    assert after_filled == before_filled + 1
    assert info["illegal"] is False
    assert reward > -0.01  # base -0.01 plus positive shaping
    assert done is False


def test_solved_detection():
    # Build an almost-solved board: final cell (8,8) is 9.
    solved = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 0],  # last cell empty
        ],
        dtype=np.int8,
    )

    env = SudokuEnv(initial_board=solved)
    env.reset()

    # Correct final move: put 9 at (8, 8)
    action = env.encode_action(row=8, col=8, digit=9)
    obs, reward, done, info = env.step(action)

    assert obs[8, 8] == 9
    assert info["solved"] is True
    assert done is True
    assert reward > 0.9  # base -0.01 + shaping + solve bonus

