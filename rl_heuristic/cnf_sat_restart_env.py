import gymnasium as gym
from gymnasium import spaces
import numpy as np 

class SatEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 20}

    def __init__(self):
        super().__init__()
        self.nb_variables =3
        self.action_space = spaces.Discrete(self.nb_variables*2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3**self.nb_variables, 3), dtype=np.int8)
        self.step_count = 0
        self.alpha = 1
        self.max_steps = 8
        self.max_solution = 10
        self.max_conflict = 9
        self.reset()

    def step(self, action):
        self.step_count += 1
        if action not in self.valid_action():
            return self.observation, -100, True, False, {"message": "Variable already assigned"}    
        
        index = (action // 2)
        sign = 1 if action % 2 == 0 else -1
        updated = []
        is_conflict = False
        is_solution = False
        nb_simplified_clause = 0
        
        for clause in self.observation:
            clause = np.array(clause)
            if clause[index] == sign:
                updated.append(np.zeros(self.nb_variables, dtype=int))
                nb_simplified_clause += 1
            elif clause[index] == -sign:
                clause[index] = 0
                if np.all(clause == 0):
                    is_conflict = True
                updated.append(clause)
            else:
                updated.append(clause)

        self.observation = np.array(updated)

        if is_conflict:
            return self.observation, self.max_conflict / self.step_count, True, False, {"message": "Conflict found","steps":self.step_count}
        if is_solution:
            return self.observation, self.max_solution / self.step_count, True, False, {"message": "Solution found","steps":self.step_count}
        
        if self.step_count >= self.max_steps:
            return self.observation, -50, True, False, {"message": "Timeout"}
        
        reward = -self.alpha + self.alpha * nb_simplified_clause
        return self.observation, reward, False, False ,{}

    def valid_action(self):
        valid_action = []
        for clause in self.observation:
            for i, v in enumerate(clause):
                if v != 0:
                    valid_action.extend([2 * i, 2 * i + 1])
        return list(set(valid_action))

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.observation = self.observation_space.sample()
        return self.observation, {}

    def render(self):
        pass

    def close(self):
        pass