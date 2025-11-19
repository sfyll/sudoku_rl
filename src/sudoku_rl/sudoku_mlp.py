import torch
import torch.nn as nn
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl

class SudokuMLP(nn.Module):
    """
    Simple feedforward Actor-Critic policy for Sudoku:
    - Input: flattened 9x9 board (81,)
    - Output:
        - logits over 729 actions (row, col, digit)
        - scalar value estimate V(s)
    """

    def __init__(self, env):
        super().__init__()

        obs_dim = env.single_observation_space.shape[0]   # 81
        act_dim = env.single_action_space.n               # 729

        # Shared trunk: feature extractor
        self.net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        # Policy head: maps shared features -> action logits
        self.action_head = nn.Linear(256, act_dim)

        # Value head: maps shared features -> scalar state value
        self.value_head = nn.Linear(256, 1)

    def forward_eval(self, observations, state=None):
        """
        Evaluation / rollout forward pass.
        observations: Tensor [batch_size, 81]
        returns:
            logits: [batch_size, 729]
            values: [batch_size, 1]
        """
        # Ensure 2D [B, 81]
        x = observations.view(observations.shape[0], -1).float()
        hidden = self.net(x)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    # Puffer likes .forward to exist; we just alias to forward_eval.
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


if __name__ == "__main__":
    # Minimal sanity test: shapes + sampling an action
    import numpy as np
    from .env_puffer import SudokuPufferEnv

    env = SudokuPufferEnv(difficulty="super-easy")
    policy = SudokuMLP(env)

    obs, infos = env.reset(seed=0)          # obs: [1, 81] np.ndarray
    obs_t = torch.as_tensor(obs, dtype=torch.float32)

    logits, values = policy.forward_eval(obs_t)

    print("logits shape:", logits.shape)    # expect [1, 729]
    print("values shape:", values.shape)    # expect [1, 1]

    # Sample an action from the policy distribution
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    print("sampled action:", action.item())

    # Step env once with the sampled action, just to see it runs
    next_obs, rew, done, trunc, infos = env.step(action.detach().cpu().numpy().reshape(1,))
    print("reward:", float(rew[0]), "done:", bool(done[0]))
