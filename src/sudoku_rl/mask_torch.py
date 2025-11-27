from __future__ import annotations

import torch


_block_idx = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8]], dtype=torch.int64)
_digit_bits = torch.tensor([1 << i for i in range(9)], dtype=torch.int64)


def legal_action_mask_torch(observations: torch.Tensor) -> torch.Tensor:
    """Bitset-based legal action mask (no one-hot), shape (B, 729)."""
    if observations.dim() == 2:
        board = observations.view(observations.shape[0], 9, 9)
    else:
        board = observations

    board = board.to(dtype=torch.int64)
    device = board.device
    B = board.shape[0]

    bits = _digit_bits.to(device)
    # Build one-hot per digit (B,9,9,9) via broadcasting equality against 1..9
    digits = torch.arange(1, 10, device=device, dtype=torch.int64).view(1, 1, 1, 9)
    present = (board.unsqueeze(-1) == digits)  # B,9,9,9

    row_mask = (present.any(dim=2) * bits).sum(dim=2)   # B,9
    col_mask = (present.any(dim=1) * bits).sum(dim=2)   # B,9

    blocks = present.view(B, 3, 3, 3, 3, 9)
    block_mask = (blocks.any(dim=(2, 4)) * bits).sum(dim=3).reshape(B, 9)  # B,9

    block_idx = _block_idx.to(device)
    block_grid = block_mask[:, block_idx]  # B,9,9

    union = row_mask[:, :, None] | col_mask[:, None, :] | block_grid
    allowed_bits = (~union) & 0x1FF
    empty = board == 0
    allowed_bits = allowed_bits * empty  # zero out filled cells

    mask = (allowed_bits[..., None] & bits) != 0  # B,9,9,9
    mask = mask.view(B, -1)

    none_legal = ~mask.any(dim=1)
    if none_legal.any():
        mask[none_legal] = True
    return mask
