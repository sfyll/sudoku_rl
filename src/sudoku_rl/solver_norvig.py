"""
Deterministic Sudoku solver with trace (Norvig-style search + constraint propagation).
Returns a sequence of board states capturing all forced-deduction cascades and decisions.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

# --- Norvig primitives -------------------------------------------------------
rows = "ABCDEFGHI"
cols = "123456789"


def cross(A, B):
    return [a + b for a in A for b in B]


squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ("ABC", "DEF", "GHI") for cs in ("123", "456", "789")])
units = {s: [u for u in unitlist if s in u] for s in squares}
peers = {s: set(sum(units[s], [])) - {s} for s in squares}


def grid_values(grid: str) -> Dict[str, str]:
    """Convert grid string into dict of {square: char} with '.' for empties."""
    assert len(grid) == 81
    return {s: grid[i] for i, s in enumerate(squares)}


def parse_grid(grid: str, states: List[np.ndarray]) -> Optional[Dict[str, str]]:
    """Parse grid into a dict of possible values, applying all assignments.

    states: list to append intermediate board snapshots (np.ndarray 9x9) after every assignment.
    Returns values dict or None on contradiction.
    """
    values = {s: cols for s in squares}  # each square can be any digit initially
    for s, d in grid_values(grid).items():
        if d in cols:
            if not assign(values, s, d, states):
                return None
    return values


def assign(values: Dict[str, str], s: str, d: str, states: List[np.ndarray]) -> Optional[Dict[str, str]]:
    """Assign digit d to square s and propagate. Return values or None if contradiction."""
    other = values[s].replace(d, "")
    if all(eliminate(values, s, d2, states) for d2 in other):
        return values
    return None


def eliminate(values: Dict[str, str], s: str, d: str, states: List[np.ndarray]) -> bool:
    """Eliminate digit d from square s. Record state when square becomes single."""
    if d not in values[s]:
        return True
    values[s] = values[s].replace(d, "")
    # Contradiction: no digits left
    if len(values[s]) == 0:
        return False
    # If only one value left, eliminate it from peers
    if len(values[s]) == 1:
        d2 = values[s]
        # Record snapshot after a forced assignment
        states.append(values_to_board(values))
        if not all(eliminate(values, p, d2, states) for p in peers[s]):
            return False
    # If a unit has only one place for digit d, assign it there
    for u in units[s]:
        dplaces = [sq for sq in u if d in values[sq]]
        if len(dplaces) == 0:
            return False
        if len(dplaces) == 1:
            if not assign(values, dplaces[0], d, states):
                return False
    return True


def search(values: Dict[str, str], states: List[np.ndarray]) -> Optional[Dict[str, str]]:
    """Depth-first search with deterministic branching (fewest possibilities, lexicographic)."""
    if values is None:
        return None
    # solved
    if all(len(values[s]) == 1 for s in squares):
        return values
    # choose unfilled square with fewest possibilities
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    # deterministic ordering of guesses
    for d in sorted(values[s]):
        new_states = states  # share to keep full trace
        new_values = dict(values)
        # on guess, append state after assignment
        attempt = assign(new_values, s, d, new_states)
        if attempt:
            result = search(attempt, new_states)
            if result:
                return result
    return None


# --- Helpers -----------------------------------------------------------------

def values_to_board(values: Dict[str, str]) -> np.ndarray:
    arr = np.zeros((9, 9), dtype=np.int8)
    for idx, s in enumerate(squares):
        r, c = divmod(idx, 9)
        v = values[s]
        if len(v) == 1 and v in cols:
            arr[r, c] = int(v)
        else:
            arr[r, c] = 0
    return arr


def board_to_grid(board: np.ndarray) -> str:
    flat = board.reshape(-1)
    chars = [str(int(x)) if x != 0 else '.' for x in flat]
    return ''.join(chars)


def solve_with_trace(board: np.ndarray) -> Optional[List[np.ndarray]]:
    """Solve puzzle and return list of board states (including initial and solution).

    Uses Norvig solver with deterministic branching. States include all forced assignments;
    remaining unsolved cells are zeros until they become singletons.
    """
    grid = board_to_grid(board)
    states: List[np.ndarray] = [np.array(board, dtype=np.int8, copy=True)]
    values = parse_grid(grid, states)
    values = search(values, states)
    if values is None:
        return None
    solved = values_to_board(values)
    if not np.array_equal(states[-1], solved):
        states.append(solved)
    return states


__all__ = ["solve_with_trace"]
