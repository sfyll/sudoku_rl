import numpy as np

import pytest

from sudoku_rl import SudokuEnv
from sudoku_rl.puzzle import sample_puzzle, supported_bins
from sudoku_rl.env import legal_action_mask, count_violations


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


def make_simple_solution():
    """
    Fully solved grid consistent with make_simple_puzzle().
    """
    return np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ],
        dtype=np.int8,
    )


def test_reset_returns_initial_board_copy():
    puzzle = make_simple_puzzle()
    solution = make_simple_solution()
    env = SudokuEnv(initial_board=puzzle, solution_board=solution)
    obs = env.reset()

    # Same values
    assert np.array_equal(obs, puzzle)

    # But not the same object
    obs[0, 0] = 0
    assert env.board[0, 0] == 5


def test_illegal_move_penalized_and_not_applied():
    puzzle = make_simple_puzzle()
    solution = make_simple_solution()
    env = SudokuEnv(initial_board=puzzle, solution_board=solution)
    env.reset()

    # Try to place a 5 in same row, which is illegal.
    action = env.encode_action(row=0, col=2, digit=5)
    obs, reward, done, info = env.step(action)

    # Board must be unchanged at that location.
    assert obs[0, 2] == 0
    assert info["illegal"] is True
    assert reward == 0.0  # no progress signal for illegal
    assert np.isclose(info["delta_F"], 0.0)
    assert done is False  # we don't terminate on illegal moves


def test_valid_move_fills_cell_and_gives_entropy_reward():
    puzzle = make_simple_puzzle()
    solution = make_simple_solution()
    env = SudokuEnv(initial_board=puzzle, solution_board=solution)
    obs0 = env.reset()

    before_filled = int(np.count_nonzero(obs0))
    # Legal move: place the correct digit according to the solution.
    action = env.encode_action(row=0, col=2, digit=int(solution[0, 2]))
    obs, reward, done, info = env.step(action)

    after_filled = int(np.count_nonzero(obs))

    assert obs[0, 2] == solution[0, 2]
    assert after_filled == before_filled + 1
    assert info["illegal"] is False
    assert reward >= 0.0
    assert info["delta_F"] >= 0.0
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

    solution = make_simple_solution()
    env = SudokuEnv(initial_board=solved, solution_board=solution)
    env.reset()

    # Correct final move: put 9 at (8, 8)
    action = env.encode_action(row=8, col=8, digit=9)
    obs, reward, done, info = env.step(action)

    assert obs[8, 8] == 9
    assert info["solved"] is True
    assert done is True
    assert reward >= SudokuEnv.SOLVE_BONUS


def test_wrong_digit_ends_episode():
    """
    Picking a locally legal but wrong digit should end the episode immediately.
    """
    board, solution = sample_puzzle(bin_label=supported_bins()[0], seed=0, return_solution=True)
    env = SudokuEnv(initial_board=board, solution_board=solution)
    env.reset()

    legal_mask = legal_action_mask(board).reshape(9, 9, 9)
    target = None
    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                continue
            digits = np.flatnonzero(legal_mask[r, c])
            if len(digits) > 1:
                correct = int(solution[r, c])
                wrong_candidates = [d + 1 for d in digits if (d + 1) != correct]
                if wrong_candidates:
                    target = (r, c, wrong_candidates[0], correct)
                    break
        if target:
            break
    assert target, "need a cell with multiple legal digits to test wrong-solution termination"
    row, col, wrong_digit, correct_digit = target

    action = env.encode_action(row=row, col=col, digit=wrong_digit)
    _, reward, done, info = env.step(action)

    assert info["illegal"] is False
    assert info["wrong_digit"] is True
    assert done is True
    assert np.isclose(reward, -SudokuEnv.WRONG_DIGIT_PENALTY)


def test_wrong_digit_against_solution_is_penalized():
    board, solution = sample_puzzle(bin_label=supported_bins()[0], seed=0, return_solution=True)
    env = SudokuEnv(initial_board=board, solution_board=solution, terminate_on_wrong_digit=False)
    env.reset()

    # Find an empty cell with >1 locally legal digit, pick a legal-but-wrong digit.
    legal_mask = legal_action_mask(board).reshape(9, 9, 9)
    target = None
    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                continue
            digits = np.flatnonzero(legal_mask[r, c])
            if len(digits) > 1:
                correct_digit = int(solution[r, c])
                wrong_candidates = [d + 1 for d in digits if (d + 1) != correct_digit]
                if wrong_candidates:
                    target = (r, c, wrong_candidates[0], correct_digit)
                    break
        if target:
            break
    assert target, "need a cell with multiple legal digits to test wrong-solution penalty"
    row, col, wrong_digit, correct_digit = target

    action = env.encode_action(row=row, col=col, digit=wrong_digit)
    obs, reward, done, info = env.step(action)

    assert info["illegal"] is False
    assert info["wrong_digit"] is True
    assert done is False  # termination disabled
    assert obs[row, col] == 0  # board unchanged on wrong-digit
    assert np.isclose(reward, -SudokuEnv.WRONG_DIGIT_PENALTY)


def test_env_handles_dataset_puzzle():
    # pick a mid bin so puzzles have enough blanks
    board, solution = sample_puzzle(bin_label=supported_bins()[1], seed=0, return_solution=True)
    env = SudokuEnv(initial_board=board, solution_board=solution)

    obs = env.reset()

    assert np.array_equal(obs, board)
    assert obs.dtype == np.int8


def test_legal_action_mask_matches_env_rules():
    puzzle = make_simple_puzzle()
    solution = make_simple_solution()
    mask = legal_action_mask(puzzle)

    env = SudokuEnv(initial_board=puzzle, solution_board=solution)
    env.reset()

    legal_action = env.encode_action(row=0, col=2, digit=1)
    illegal_action = env.encode_action(row=0, col=2, digit=5)

    assert mask[legal_action]
    assert not mask[illegal_action]


def test_count_violations_empty_and_valid_board():
    empty = np.zeros((9, 9), dtype=np.int8)
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
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ],
        dtype=np.int8,
    )

    assert count_violations(empty) == 0
    assert count_violations(solved) == 0


@pytest.mark.parametrize(
    "board, expected",
    [
        # Single duplicate in a row (no column/block conflict)
        (
            np.array(
                [
                    [5, 0, 0, 0, 5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            1,
        ),
        # Single duplicate in a column (no row/block conflict)
        (
            np.array(
                [
                    [4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            1,
        ),
        # Single duplicate within a 3x3 block (no row/column conflict)
        (
            np.array(
                [
                    [7, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            1,
        ),
    ],
)
def test_count_violations_single_conflict(board, expected):
    assert count_violations(board) == expected


def test_count_violations_sums_multiple_conflicts():
    board = np.array(
        [
            [5, 6, 0, 0, 5, 0, 0, 0, 0],  # row duplicate: two 5s
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 0, 7, 0, 0, 0, 0],  # column duplicate at col 1
            [0, 0, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )

    # Expected: row duplicate (1) + column duplicate (1) + block duplicate of 7s in center block (1) = 3
    assert count_violations(board) == 3
