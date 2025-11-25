from __future__ import annotations
from typing import Counter

import numpy as np
import pufferlib
import gymnasium

from .env import SudokuEnv
from .puzzle import sample_puzzle


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
        bin_label: str | None = None,
        buf=None,
        seed: int = 0,
        max_steps: int = 10_000,
        initial_board=None,
        terminate_on_wrong_digit: bool = True,
        prev_mix_ratio: float = 0.3,
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
        self.prev_mix_ratio = prev_mix_ratio
        if initial_board is None:
            board, solution = sample_puzzle(
                bin_label=bin_label,
                seed=seed,
                return_solution=True,
                prev_mix_ratio=prev_mix_ratio,
            )
        elif isinstance(initial_board, tuple) and len(initial_board) == 2:
            board, solution = initial_board
        else:
            raise ValueError("Provide (board, solution) tuple for initial_board, or leave it None to sample.")

        self.env = SudokuEnv(
            initial_board=board,
            solution_board=solution,
            max_steps=max_steps,
            terminate_on_wrong_digit=terminate_on_wrong_digit,
        )
        self.bin_label = bin_label
        self._done = False

        # ---- Let Puffer allocate buffers ----
        super().__init__(buf)

    # ---- Required API: reset/step/render/close ----
    def reset(self, seed: int | None = None):
        """Reset SudokuEnv and write into Puffer buffers."""
        if seed is None:
            seed = self._seed

        board, solution = sample_puzzle(
            bin_label=self.bin_label,
            return_solution=True,
            prev_mix_ratio=self.prev_mix_ratio,
        )

        board = self.env.reset(seed=seed, initial_board=board, solution_board=solution)
        self._done = False

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

        # If the episode ended, signal termination on this step and queue up
        # the next episode so the runner never stalls on a solved puzzle.
        if done:
            self._done = True
            self.terminals[0] = True
            self.truncations[0] = False
            self.masks[0] = False
            self.rewards[0] = float(reward)
            # Add episode-level stats to info; PufferLib will average these
            # across steps since the last log.
            info = info.copy()
            info["steps_in_episode"] = self.env.steps
            info["solved_episode"] = 1.0 if info.get("solved") else 0.0
            info["steps_per_empty"] = self.env.steps / max(1, self.env.initial_empties)
            infos = [info]

            # Leave observations as-is; Serial backend will call reset
            # before the next step when it sees env.done
            return self.observations, self.rewards, self.terminals, self.truncations, infos

        # Write back into Puffer buffers (in-place)
        self.observations[0, :] = board.reshape(-1)
        self.rewards[0] = float(reward)
        self.terminals[0] = bool(done)
        self._done = bool(done)

        # We use "done" as terminal, not truncation, for now. Difference is one is task-based (fatal error, i.e. fell of map, agent died or puzzle solved), the other is for technical reason (max step allowed).
        self.truncations[0] = False
        self.masks[0] = not done

        infos = [info]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def render(self, mode=None):
        """
        Render the board.
        - "ansi" -> ASCII grid
        - "rgb_array" -> uint8 HxWx3 image for video logging
        - default -> numpy board copy
        """
        mode = mode or self.render_mode
        board = self.env.board  # (9, 9)

        if mode == "ansi":
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

        if mode == "rgb_array":
            cell = 32
            line = 2
            size = cell * 9 + line * 8
            img = np.full((size, size, 3), 255, dtype=np.uint8)

            # Grid lines
            for i in range(8):
                x = (i + 1) * cell + i * line
                img[:, x:x+line] = 0
                img[x:x+line, :] = 0

            # 3x3 block lines thicker
            thick = 4
            for i in [3, 6]:
                x = i * cell + (i - 1) * line
                img[:, x:x+thick] = 0
                img[x:x+thick, :] = 0

            # Color map for digits; empty = light gray
            palette = np.array([
                [230, 230, 230],
                [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
                [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34]
            ], dtype=np.uint8)

            for r in range(9):
                for c in range(9):
                    val = board[r, c]
                    color = palette[val]
                    y0 = r * cell + r * line
                    x0 = c * cell + c * line
                    img[y0:y0+cell, x0:x0+cell] = color

            return img

        # Fallback: numeric board copy
        return board.copy()

    def close(self):
        pass

    # Expose done flag so pufferlib Serial can trigger a reset
    @property
    def done(self):
        return self._done


if __name__ == "__main__":
    # Standalone random-rollout debug for SudokuPufferEnv
    import time
    import numpy as np
    from collections import Counter

    seed = 0  # easiest puzzle
    env = SudokuPufferEnv(bin_label="zeros_04_07")
    obs, info = env.reset(seed=seed)
    steps = 0

    CACHE = 1024
    # Pre-generate random actions in [0, 9*9*9)
    actions = np.random.randint(
        0,
        env.single_action_space.n,
        size=(CACHE, 1),
        dtype=np.int64,
    )

    rewards = []
    reasons = Counter()
    episodes = 0

    start = time.time()
    DURATION = 5.0  # seconds

    while time.time() - start < DURATION:
        # Optional: uncomment if you want to inspect the board
        # print("board:\n", env.env.board)

        action = actions[steps % CACHE]
        obs, rew, term, trunc, infos = env.step(action)
        steps += 1

        # rew is shape (1,) from Puffer; store scalar
        rewards.append(float(rew[0]))

        info0 = infos[0]
        if term[0] or trunc[0]:
            if info0.get("solved"):
                reason = "solved"
            elif info0.get("timeout"):
                reason = "timeout"
            else:
                reason = "other"
            reasons[reason] += 1
            episodes += 1
            obs, info = env.reset()

    elapsed = time.time() - start
    print("SudokuPufferEnv SPS:", int(steps / elapsed))
    print("Episodes:", episodes)
    print(f"Succes rate: {100.0 * reasons['solved'] / episodes:.1f}%")
    print("Reward mean/std:", np.mean(rewards), np.std(rewards))
    print("Done reasons:", reasons)
    print("Last board:")
    print(env.render())
