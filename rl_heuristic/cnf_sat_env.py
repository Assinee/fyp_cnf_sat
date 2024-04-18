import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(40)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(91, 20), dtype=np.int8)
        self.step_count = 0
        self.alpha = 0.1
        self.max_solution = 10
        self.max_conflict = 8
        self.reset()
    
    def step(self, action):
        self.step_count += 1
        
        # check if valide action
        if action not in self.valid_action():
            return self.observation, -100, True, {"message": "Variable already assigned"}    
        
        # update observation/formula
        index = action // 2
        sign = 1 if action % 2 == 0 else -1
        updated = []
        is_conflict = False
        is_solution = False
        nb_simplified_clause = 0
        for clause in self.observation:
            if clause[index] == sign:
                updated.append([0] * 20)
                nb_simplified_clause += 1
                if self.observation==[[[0] * 20]*3**20]:
                    is_solution = True
            elif clause[index] == -sign:
                clause[index] = 0
                if all(v == 0 for v in clause):
                    is_conflict = True
                updated.append(clause)
            else:
                updated.append(clause)

        self.observation = np.array(updated)

        if is_conflict:
            return self.observation, self.max_conflict / self.step_count, True, {"message": "Conflict found"}
        if is_solution:
            return self.observation, self.max_solution / self.step_count, True, {"message": "Solution found"}

        reward = -self.alpha + self.alpha * nb_simplified_clause
        return self.observation, reward, False, {}
    
    def valid_action(self):
        unassigned_variable = []
        valid_action = []
        for clause in self.observation:
            for i, v in enumerate(clause):
                if v != 0 and i not in unassigned_variable:
                    unassigned_variable.extend([i, -i])
        
        for i in range(0, len(unassigned_variable), 2):
            valid_action.append(2 * unassigned_variable[i])
            valid_action.append(2 * unassigned_variable[i] + 1)
        return valid_action

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.observation = self.observation_space.sample()  # Generate a new observation
        return self.observation,{}

    def render(self):
        pass

    def close(self):
        pass
