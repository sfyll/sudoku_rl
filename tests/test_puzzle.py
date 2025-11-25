import numpy as np

from sudoku_rl.puzzle import get_puzzle_pool, sample_puzzle, supported_bins


def test_get_puzzle_pool_reads_processed_csv():
    pool = get_puzzle_pool(supported_bins()[0])

    assert len(pool) > 0
    def puzzle_str(entry):
        return entry["puzzle"] if isinstance(entry, dict) else entry

    assert all(len(puzzle_str(p)) == 81 for p in pool)
    assert all(ch.isdigit() for puzzle in pool for ch in puzzle_str(puzzle))


def test_sample_puzzle_returns_board_ready_for_env():
    # Use the hardest available bin to ensure we still load correctly
    board = sample_puzzle(supported_bins()[-1], seed=0)

    assert board.shape == (9, 9)
    assert board.dtype == np.int8
    assert np.all((board >= 0) & (board <= 9))
    # Ensure puzzles contain at least one empty cell for learning signal.
    assert np.count_nonzero(board == 0) > 0
