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
        self.action_space = spaces.Discrete(self.num_variables * 2)  # Choose a variable and assign true or false
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_variables, 3), dtype=np.float32)  # One-hot encoded state
        
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
        variable_index = action // 2
        variable_value = (action % 2) * 2 - 1  # Map action to -1 or 1

        # Check if the variable is already assigned a non-zero value
        if self.assignment[variable_index] != 0:
            # Infinite penalty for reassigning an already assigned variable
            return self.create_observation(), -10000, True, False,{"message": "Variable already assigned"}

        # Update the assignment
        self.assignment[variable_index] = variable_value

        # Simplify the formula based on the current assignments
        new_formula = self.formula[:]
        for i, value in enumerate(self.assignment):
            if value != 0:
                var = value * (self.variables[i])  # Adjusting the index according to variable list
                new_formula = [clause for clause in new_formula if var not in clause]
                new_formula = [[x for x in clause if x != -var] for clause in new_formula]

        self.formula = new_formula
        
        reward = -self.alpha  # Default small penalty
        if self.step_count !=0:
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

        # Update observation
        observation = self.create_observation()
        return observation, reward, done, False ,info

    
    def create_observation(self):
        """Create the observation from the current assignment state."""
        obs = np.zeros((self.num_variables, 3))
        for idx, value in enumerate(self.assignment):
            if value == 0:
                obs[idx, 0] = 1  # Unassigned
            elif value == 1:
                obs[idx, 1] = 1  # True
            elif value == -1:
                obs[idx, 2] = 1  # False
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.formula = list(self.original_formula)  # Reset formula to initial state
        self.assignment = np.zeros_like(self.original_assigned_variables)
        self.step_count = 0  # Reset step count
        initial_observation = self.create_observation()  # Ensure this matches the observation space shape
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
