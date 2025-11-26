from __future__ import annotations

import numpy as np

FULL_MASK = (1 << 9) - 1  # bits 0..8 correspond to digits 1..9


def _bit_for_digit(digit: int) -> int:
    return 1 << (digit - 1)


def legal_action_mask_fast(board: np.ndarray) -> np.ndarray:
    """Vectorized legal action mask for a 9x9 Sudoku board.

    Returns a flat bool mask of shape (729,). Falls back to the slower path
    if the board shape is unexpected.
    """
    if board.shape != (9, 9):
        raise ValueError(f"board must be 9x9, got {board.shape}")

    board = np.asarray(board, dtype=np.int8)

    row_mask = np.zeros(9, dtype=np.uint16)
    col_mask = np.zeros(9, dtype=np.uint16)
    block_mask = np.zeros(9, dtype=np.uint16)

    # Accumulate used digits as bitsets
    nonzero = board != 0
    rows, cols = np.nonzero(nonzero)
    vals = board[rows, cols]
    bits = np.left_shift(np.uint16(1), vals.astype(np.int32) - 1).astype(np.uint16)
    for r, c, b in zip(rows, cols, bits):
        row_mask[r] |= b
        col_mask[c] |= b
        block_idx = (r // 3) * 3 + (c // 3)
        block_mask[block_idx] |= b

    # For quick lookup, map each cell to its block mask
    block_grid = np.empty((9, 9), dtype=np.uint16)
    for r in range(9):
        for c in range(9):
            block_grid[r, c] = block_mask[(r // 3) * 3 + (c // 3)]

    mask = np.zeros(9 * 9 * 9, dtype=bool)

    empties = board == 0
    if not np.any(empties):
        return mask  # No legal moves if board full

    for d in range(1, 10):
        bit = _bit_for_digit(d)
        allowed_cells = (
            empties
            & ((row_mask[:, None] & bit) == 0)
            & ((col_mask[None, :] & bit) == 0)
            & ((block_grid & bit) == 0)
        )
        if not allowed_cells.any():
            continue
        rows_d, cols_d = np.nonzero(allowed_cells)
        action_indices = rows_d * (9 * 9) + cols_d * 9 + (d - 1)
        mask[action_indices] = True

    if not mask.any():
        mask[:] = True  # preserve old fallback behaviour

    return mask
