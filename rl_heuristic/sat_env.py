import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, formula, assigned_variables ,max_solution=10, max_conflict=8 , alpha=0.1):
        super().__init__()
        self.original_formula = formula
        self.original_assigned_variables = assigned_variables

        self.formula = list(formula)  
        self.variables = self.get_all_variables(formula)
        self.num_variables = len(self.variables)

        self.action_space = spaces.Discrete(self.num_variables * 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_variables,), dtype=np.float32)

        # self.assignment = []
        self.assignment = assigned_variables 
        self.step_count = 0  # To calculate depth for the reward function

        # Reward function parameters
        self.max_solution = max_solution
        self.max_conflict = max_conflict
        self.alpha = alpha

    def get_all_variables(self, formula):
        return sorted(list(set(abs(num) for sublist in formula for num in sublist)))   
    
    def step(self, action):
        # Increment step count
        self.step_count += 1
        variable_index = action // 2
        variable_value = (action % 2) * 2 - 1
        # self.assignment.append(variable)

        self.assignment[variable_index] = variable_value

        new_formula=self.formula
        assigned_variables=[]
        for i in range(len(self.assignment)):
            if self.assignment[i] != 0:
                assigned_variables.append(self.assignment[i]*(i+1))
                
        for variable in assigned_variables:
            new_formula = [clause for clause in new_formula if variable not in clause]
            new_formula = [[x for x in sublist if x != -variable] for sublist in new_formula]

        self.formula = new_formula
        
        if new_formula == []: #solution found
            reward = self.step_count * self.alpha + self.max_solution * 1/self.step_count
        if  any(len(clause) == 0 for clause in new_formula): #found a conflict
            reward = self.step_count * self.alpha + self.max_conflict * 1/self.step_count
        else:
            reward = -self.alpha  # Small penalty for each step

        # Check for conflicts or an empty formula (solved)
        done = (new_formula == [] or  any(len(clause) == 0 for clause in new_formula))
        
        info = {"formula": self.formula, "assigned variables": self.assignment , "find a conflict": any(len(clause) == 0 for clause in new_formula)}

        return np.array(self.assignment, dtype=np.float32), reward, done, False, info

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.formula = list(self.original_formula)  # Reset formula to initial state
        self.assignment = list(self.original_assigned_variables)
        self.step_count = 0  # Reset step count
        initial_observation = np.zeros(self.num_variables, dtype=np.float32)  # Ensure this matches the observation space shape
        return initial_observation,{}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current assignment: {self.assignment}")
            print(f"Remaining formula: {self.formula}")

    def close(self):
        pass