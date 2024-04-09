import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, formula, R_max=10, alpha=0.1):
        super().__init__()
        self.original_formula = formula
        self.formula = list(formula)  
        self.variables = self.get_all_variables(formula)
        self.num_variables = len(self.variables)

        self.action_space = spaces.Discrete(self.num_variables * 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_variables,), dtype=np.float32)

        self.assignment = np.zeros(self.num_variables, dtype=int)
        self.step_count = 0  # To calculate depth for the reward function

        # Reward function parameters
        self.R_max = R_max
        self.alpha = alpha

    def get_all_variables(self, formula):
        return sorted(list(set(abs(num) for sublist in formula for num in sublist)))

    def step(self, action):
        # Increment step count
        self.step_count += 1

        # Decode action into variable index and assignment value
        var_index = action % self.num_variables
        assign_value = 1 if action >= self.num_variables else -1
        variable = self.variables[var_index]

        # Update formula based on the assignment
        self.apply_assignment(variable, assign_value)
        self.assignment[var_index] = assign_value

        # Check for conflicts or an empty formula (solved)
        done = len(self.formula) == 0 or any(len(clause) == 0 for clause in self.formula)
        
        # Calculate reward based on the depth
        if len(self.formula) == 0:  # Solution found
            reward = self.R_max - self.alpha * self.step_count
        elif done:  # Unsatisfiable state reached
            reward = -self.R_max  # Maximum penalty for failure
        else:
            reward = -self.alpha  # Small penalty for each step

        info = {"formula": self.formula}

        return np.array(self.assignment, dtype=np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.formula = list(self.original_formula)  # Reset formula to initial state
        self.assignment = np.zeros(self.num_variables, dtype=int)  # Reset assignments
        self.step_count = 0  # Reset step count
        return np.array(self.assignment, dtype=np.float32), {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current assignment: {self.assignment}")
            print(f"Remaining formula: {self.formula}")

    def close(self):
        pass

    def apply_assignment(self, variable, value):
        new_formula = []
        for clause in self.formula:
            if variable in clause and value == 1:
                continue  # Clause is satisfied, remove it
            if -variable in clause and value == -1:
                continue  # Clause is satisfied, remove it
            new_clause = [x for x in clause if x != variable and x != -variable]  # Remove variable and its negation
            new_formula.append(new_clause)
        self.formula = new_formula