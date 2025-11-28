import numpy as np

from sudoku_rl.env import legal_action_mask


def _random_board(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    board = np.zeros((9, 9), dtype=np.int8)
    # Fill ~20 random cells with valid digits ignoring global consistency
    cells = rng.choice(81, size=20, replace=False)
    for idx in cells:
        r, c = divmod(idx, 9)
        board[r, c] = int(rng.integers(1, 10))
    return board

