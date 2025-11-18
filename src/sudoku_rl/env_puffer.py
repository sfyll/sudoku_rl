from __future__ import annotations

import numpy as np
import gym 
import pufferlib

from .env import SudokuEnv


class SudokuPufferEnv(pufferlib.PufferEnv):
    """
    Puffer-native wrapper around our Python SudokuEnv (single-agent).

    - Observation: flattened 9x9 Sudoku board, shape (81,), values 0..9
    - Action: Discrete(729) encoding (row, col, digit) via SudokuEnv.step
    """

    # Required keyword arguments: render_mode, buf, seed
    def __init__(
        self,
        render_mode: str = "ansi",
        buf=None,
        seed: int = 0,
        max_steps: int = 200,
        initial_board=None,
    ):
        # ---- Required attributes BEFORE super().__init__ ----
        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=9,
            shape=(81,),
            dtype=np.int8,
        )
        # 9 rows * 9 cols * 9 digits = 729 actions
        self.single_action_space = gymnasium.spaces.Discrete(9 * 9 * 9)
        self.render_mode = render_mode
        self.num_agents = 1  # single-agent

        # ---- Our own logical env (Phase 2) ----
        self._seed = seed
        self.max_steps = max_steps
        self.env = SudokuEnv(initial_board=initial_board, max_steps=max_steps)

        # ---- Let Puffer allocate buffers ----
        super().__init__(buf)

    # ---- Required API: reset/step/render/close ----

    def reset(self, seed: int | None = None):
        """Reset SudokuEnv and write into Puffer buffers."""
        if seed is None:
            seed = self._seed

        board = self.env.reset(seed=seed)

        # observations shape: (num_agents, 81) => (1, 81)
        self.observations[0, :] = board.reshape(-1)

        # reset rewards/done/masks
        self.rewards[0] = 0.0
        self.terminals[0] = False
        self.truncations[0] = False
        self.masks[0] = True

        infos = [{}]  # one dict per agent
        return self.observations, infos

    def step(self, actions):
        """
        actions: np.ndarray with shape (num_agents, ...) = (1,)
        Contains a Discrete(729) action index.
        """
        atn = int(actions[0])

        # Delegate all Sudoku logic to our Phase 2 env
        board, reward, done, info = self.env.step(atn)

        # Write back into Puffer buffers (in-place)
        self.observations[0, :] = board.reshape(-1)
        self.rewards[0] = float(reward)
        self.terminals[0] = bool(done)

        # We use "done" as terminal, not truncation, for now. Difference is one is task-based (fatal error, i.e. fell of map, agent died or puzzle solved), the other is for technical reason (max step allowed).
        self.truncations[0] = False
        self.masks[0] = not done

        infos = [info]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self):
        """Simple ASCII Sudoku renderer for render_mode='ansi'."""
        board = self.env.board  # (9, 9) from SudokuEnv
        if self.render_mode != "ansi":
            # You could plug in a fancier renderer later
            return board.copy()

        lines = []
        for r in range(9):
            if r % 3 == 0:
                lines.append("+-------+-------+-------+\n")
            row_chars = []
            for c in range(9):
                if c % 3 == 0:
                    row_chars.append("| ")
                val = board[r, c]
                row_chars.append("." if val == 0 else str(val))
                row_chars.append(" ")
            row_chars.append("|\n")
            lines.append("".join(row_chars))
        lines.append("+-------+-------+-------+\n")
        return "".join(lines)

    def close(self):
        pass


if __name__ == "__main__":
    # Run SudokuPufferEnv standalone, like PySquared
    env = SudokuPufferEnv()
    obs, info = env.reset()
    steps = 0

    CACHE = 1024
    # Random actions (mostly illegal, just for perf testing / sanity)
    actions = np.random.randint(0, 9 * 9 * 9, (CACHE, 1))

    import time

    start = time.time()
    while time.time() - start < 5:
        obs, rew, term, trunc, info = env.step(actions[steps % CACHE])
        steps += 1

    print("SudokuPufferEnv SPS:", int(steps / (time.time() - start)))
    print("Last board:")
    print(env.render())

