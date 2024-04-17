import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, formula, assigned_variables, max_solution=10, max_conflict=8, alpha=0.1):
        super().__init__()
        self.original_formula = formula
        self.original_assigned_variables = assigned_variables
        
        self.formula = list(formula)
        self.variables = self.get_all_variables(formula)
        self.num_variables = len(self.variables)
        
        # Initialize the action space and observation space
        self.action_space = spaces.Discrete(self.num_variables * 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_variables,), dtype=np.float32)
        
        self.assignment = np.copy(assigned_variables)  # Copy to avoid modifying the original array
        self.step_count = 0
        
        # Reward function parameters
        self.max_solution = max_solution
        self.max_conflict = max_conflict
        self.alpha = alpha

    def get_all_variables(self, formula):
        """Extracts all unique variables from the formula."""
        return sorted(list(set(abs(num) for sublist in formula for num in sublist)))

    def step(self, action):
        
        """Execute one time step within the environment."""
        self.step_count += 1
        variable_index = action // 2
        variable_value = (action % 2) * 2 - 1
        # print(self.assignment)
        if self.assignment[variable_index] != 0:
            return np.array(self.assignment, dtype=np.float32),-1000, True, False, {"message": "Variable already assigned"}

        # print((variable_index+1)*variable_value)
        # Update the assignment
        self.assignment[variable_index] = variable_value

        # Simplify the formula based on the current assignments
        new_formula = self.formula[:]
        for i, value in enumerate(self.assignment):
            if value != 0:
                var = value * (i + 1)
                new_formula = [clause for clause in new_formula if var not in clause]
                new_formula = [[x for x in clause if x != -var] for clause in new_formula]

        self.formula = new_formula
        
        reward = -self.alpha  # Default small penalty
        if not new_formula:
            reward = self.max_solution / self.step_count  # Reward for finding a solution
        elif any(len(clause) == 0 for clause in new_formula):
            reward = -self.max_conflict / self.step_count  # Penalty for conflict

        done = not new_formula or any(len(clause) == 0 for clause in new_formula)
        info = {
            "formula": self.formula,
            "assigned_variables": self.assignment,
            "conflict": any(len(clause) == 0 for clause in new_formula)
        }

        return np.array(self.assignment, dtype=np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.formula = list(self.original_formula)  # Reset formula to initial state
        self.assignment = list(self.original_assigned_variables)
        self.step_count = 0  # Reset step count
        initial_observation = np.zeros(self.num_variables, dtype=np.float32)  # Ensure this matches the observation space shape
        return initial_observation,{}
        
    def render(self, mode='human'):
        """Renders the environment."""
        # if mode == 'human':
        #     print(f"Current assignment: {self.assignment}")
        #     print(f"Remaining formula: {self.formula}")
        pass

    def close(self):
        """Perform any necessary cleanup."""
        pass