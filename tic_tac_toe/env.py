import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TicTacToeEnv(gym.Env):
    """Custom Tic Tac Toe Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Action space is a discrete 9 spaces, one for each cell of the 3x3 board
        self.action_space = spaces.Discrete(9)
        # Observation space is 9 integers (one for each cell), where 0 = empty, 1 = X, 2 = O
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int64)
        self.state = None
        self.current_player = 1  # 1 for X, 2 for O

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state[action] != 0:
            return self.state, -10, True, False, {}  # Invalid move

        self.state[action] = self.current_player  # Make the move
        reward, done = self.check_game_status()

        if not done:
            self.current_player = 2 if self.current_player == 1 else 1

        return self.state.copy(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.state = np.zeros(9, dtype=np.int64)
        self.current_player = 1  # X starts
        return self.state.copy(), {}

    def render(self, mode='human'):
        print(self.state.reshape(3,3))

    def close(self):
        pass

    def check_game_status(self):
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.state[x] == self.state[y] == self.state[z] != 0:
                return 1 if self.state[x] == self.current_player else -1, True
        if all(self.state != 0):
            return 0, True  # Draw
        return 0, False  # Game continues


