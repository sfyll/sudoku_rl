import numpy as np
import torch

from sudoku_rl.env import legal_action_mask
from sudoku_rl.mask_torch import legal_action_mask_torch


def _random_board(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    board = np.zeros((9, 9), dtype=np.int8)
    cells = rng.choice(81, size=20, replace=False)
    for idx in cells:
        r, c = divmod(idx, 9)
        board[r, c] = int(rng.integers(1, 10))
    return board


def test_mask_torch_matches_numpy():
    boards = []
    for seed in range(5):
        boards.append(_random_board(seed))
    boards_np = np.stack(boards)
    boards_t = torch.from_numpy(boards_np.reshape(len(boards), -1))

    torch_mask = legal_action_mask_torch(boards_t).cpu().numpy()
    numpy_mask = np.stack([legal_action_mask(b) for b in boards])

    np.testing.assert_array_equal(torch_mask, numpy_mask)

