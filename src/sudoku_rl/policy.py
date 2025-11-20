import torch
import torch.nn as nn
import pufferlib.models

class SudokuPolicy(nn.Module):
    """
    A simple Actor-Critic policy for Sudoku.
    
    In Reinforcement Learning:
    - The **Actor** is the part of the brain that decides WHAT to do. 
      It looks at the board and outputs a probability for every possible move.
    - The **Critic** is the part of the brain that estimates HOW GOOD the current situation is.
      It helps the Actor learn by telling it if a move led to a better or worse state than expected.
    """
    def __init__(self, envs, hidden_size=256):
        super().__init__(envs)
        
        # 1. The "Encoder" - The Eyes
        # We take the raw board (81 numbers) and process it into a "hidden representation".
        # The board values are 0-9. We can treat them as continuous numbers or categories.
        # For simplicity, we'll treat them as input features.
        self.input_size = 81
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 2. The "Actor" - The Decision Maker
        # Outputs logits for every possible action (729 moves).
        # We use the action_space from the environment to know how many outputs we need.
        self.actor = nn.Linear(hidden_size, envs.single_action_space.n)

        # 3. The "Critic" - The Judge
        # Outputs a single number: the Value of the state.
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        """
        The forward pass is where the thinking happens.
        We take observations and return:
        - actions: The chosen moves (sampled from the actor's output probabilities)
        - value: The critic's judgment of the state
        """
        # Observations come in as a batch. 
        # PufferLib might give us a slightly complex structure, but for a flat box, 
        # it usually just works if we cast to float.
        
        # Ensure input is float for the neural net
        hidden = observations.float()
        
        # Pass through the encoder
        hidden = self.encoder(hidden)

        # Get the action logits (unnormalized probabilities) and value
        actions_logits = self.actor(hidden)
        value = self.critic(hidden)

        return actions_logits, value
