import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.float32)
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1
        self.done = False
        return self.board

    def step(self, action):
        row, col = action // 3, action % 3
        if self.board[row, col] != 0:
            return self.board, -10, self.done, {}  # Invalid move, penalize agent
        self.board[row, col] = self.current_player
        reward, done = self._check_winner()
        self.current_player *= -1  # Switch player
        self.done = done
        return self.board, reward, done, {}

    def _check_winner(self):
        for player in [-1, 1]:
            # Check rows and columns
            if np.any(np.sum(self.board, axis=0) == player * 3) or np.any(np.sum(self.board, axis=1) == player * 3):
                return player, True
            # Check diagonals
            if np.sum(np.diag(self.board)) == player * 3 or np.sum(np.diag(np.fliplr(self.board))) == player * 3:
                return player, True
        if np.all(self.board != 0):  # Board is full
            return 0, True
        return 0, False

    def render(self):
        for row in self.board:
            print("|".join(["X" if val == 1 else "O" if val == -1 else " " for val in row]))
            print("-----")
