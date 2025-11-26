from __future__ import annotations

import torch


# Precompute block indices for 9x9 Sudoku
_block_idx = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8]], dtype=torch.int64)


def legal_action_mask_torch(observations: torch.Tensor) -> torch.Tensor:
    """Pure-Torch legal action mask.

    Args:
        observations: shape (B, 81) or (B, 9, 9), int/float, values 0..9

    Returns:
        mask: bool tensor shape (B, 729)
    """
    if observations.dim() == 2:
        board = observations.view(observations.shape[0], 9, 9)
    else:
        board = observations

    board = board.to(dtype=torch.int64)
    B = board.shape[0]

    # One-hot digits 0..9
    digits_onehot = torch.nn.functional.one_hot(board.clamp(min=0, max=9), num_classes=10)

    # Row/col presence (exclude digit 0)
    row_has = digits_onehot.sum(dim=2) > 0            # B,9,10
    col_has = digits_onehot.sum(dim=1) > 0            # B,9,10
    row_has = row_has[:, :, 1:]                       # B,9,9
    col_has = col_has[:, :, 1:]                       # B,9,9

    # Block presence: reshape into 3x3 blocks then flatten blocks
    blocks = board.view(B, 3, 3, 3, 3)                # B, br, ir, bc, ic
    blocks = blocks.permute(0, 1, 3, 2, 4).reshape(B, 9, 3, 3)
    block_onehot = torch.nn.functional.one_hot(blocks, num_classes=10)
    block_has = block_onehot.any(dim=(2, 3))[:, :, 1:]   # B,9,9

    # Broadcast forbidden masks per cell
    empty = board == 0
    block_idx = _block_idx.to(board.device)
    block_idx_expand = block_idx.unsqueeze(0).expand(B, 9, 9)
    block_forbidden = block_has[torch.arange(B, device=board.device).view(-1, 1, 1),
                                block_idx_expand]     # B,9,9,9

    row_forbidden = row_has[:, :, None, :]            # B,9,1,9
    col_forbidden = col_has[:, None, :, :]            # B,1,9,9

    forbidden = row_forbidden | col_forbidden | block_forbidden
    allowed = (~forbidden) & empty[:, :, :, None]

    mask = allowed.view(B, -1)

    # If no legal move, fall back to all True to avoid -inf masks
    none_legal = ~mask.any(dim=1)
    if none_legal.any():
        mask[none_legal] = True
    return mask

