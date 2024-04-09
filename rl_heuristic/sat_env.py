import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter

class SatEnv(gym.Env):
    """A SAT solving environment for reinforcement learning, integrating custom SAT solving logic."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, formula):
        super().__init__()
        self.original_formula = formula
        self.formula = list(formula)  # Copy to avoid modifying the original formula
        self.variables = self.get_all_variables(formula)
        self.num_variables = 20

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_variables * 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_variables,), dtype=np.float32)

        self.assignment = []

    def get_all_variables(self, formula):
        return list(set(abs(num) for sublist in formula for num in sublist))

    def step(self, action):
        # Decode action into variable index and assignment value
        var_index = action % self.num_variables
        assign_value = 1 if action >= self.num_variables else -1
        variable = self.variables[var_index]

        # Update formula based on the assignment
        self.apply_assignment(variable, assign_value)
        self.assignment[var_index] = assign_value

        # Check for conflicts or an empty formula (solved)
        done = len(self.formula) == 0 or any(len(clause) == 0 for clause in self.formula)
        reward = 1 if len(self.formula) == 0 else -1 if done else 0
        info = {}

        # Adjusting the return statement to include `truncated`
        return np.array(self.assignment, dtype=np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.formula = list(self.original_formula)  # Reset formula to initial state
        self.assignment = [0] * self.num_variables  # Reset assignments
        return np.array(self.assignment, dtype=np.float32), {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current assignment: {self.assignment}")
            print(f"Remaining formula: {self.formula}")

    def close(self):
        # Optional: Implement any necessary cleanup
        pass

    def apply_assignment(self, variable, value):
        # Implement the assignment logic here, similar to your original approach
        new_formula = []
        for clause in self.formula:
            if variable in clause and value == 1:
                continue  # Clause is satisfied, remove it
            if -variable in clause and value == -1:
                continue  # Clause is satisfied, remove it
            new_clause = [x for x in clause if x != variable and x != -variable]  # Remove variable and its negation
            if new_clause:
                new_formula.append(new_clause)
            else:
                # This means we have a conflict because the clause would be empty, return immediately
                self.formula = [[]]  # Representing an unsatisfiable state
                return
        self.formula = new_formula
