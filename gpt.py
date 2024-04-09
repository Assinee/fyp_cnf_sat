#%%
import openai


openai.api_key = "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"
model_engine = "gpt-4"

responses = []

prompt = """let say i wan to train an agent to choose wich variable to assigne next and beat this code that take the most frequent variable

from collections import Counter import time import torch import gzip def read_cnf_file(filename): formula = [] opener = gzip.open if filename.endswith('.gz') else open with opener(filename, 'rt') as file: for line in file: if not (line.startswith('c') or line.startswith('p')): clause = [] for token in line.split(): if token != '0' and token.lstrip('-').isdigit(): clause.append(int(token)) if clause: formula.append(clause) return formula def is_a_conflict(lst): if isinstance(lst, list): for sub_lst in lst: if isinstance(sub_lst, list) and len(sub_lst) == 0: return True return False def get_most_frequent_variable(formula): flat_list_abs = [abs(num) for sublist in formula for num in sublist] flat_list = [num for sublist in formula for num in sublist] counts = Counter(flat_list_abs) variable = max(counts, key=counts.get) positive_count = flat_list.count(variable) negative_count = flat_list.count(-variable) if positive_count >= negative_count: return variable else: return -variable def solve(formula, assigned_variables=[], branch_count=0): if assigned_variables == []: new_formula = formula else: variable = assigned_variables[-1] new_formula = [clause for clause in formula if variable not in clause] new_formula = [[x for x in sublist if x != -variable] for sublist in new_formula] if new_formula == []: return (assigned_variables, branch_count) if is_a_conflict(new_formula): return (None, branch_count) new_variable = get_most_frequent_variable(new_formula) assigned_variables.append(new_variable) assigned_variables_new= assigned_variables[:] result, branch_count = solve(new_formula, assigned_variables_new, branch_count + 1) if result is not None: return (result, branch_count) var = assigned_variables.pop() assigned_variables.append(-new_variable) assigned_variables_new= assigned_variables[:] return solve(new_formula, assigned_variables_new, branch_count + 1) # Example usage: # formula = [[1, -2, 3], [-1, 3], [-1, 2, 3], [1, -2]] # formula = [[1, 2], [1, -2], [-1, 2], [-1, -2]] formula=read_cnf_file('/home/assine/fyp/dataset_fyp/uf20-01.cnf') start_time = time.time() result, branch_count = solve(formula) elapsed_time = time.time() - start_time print("Time taken:", elapsed_time, "seconds") if result is not None: print("Satisfiable. Assignment:", result) print("Numbre of variable assigned",len(result)) print(f"Number of branches searched: {branch_count}") else: print("Unsatisfiable.")

how can i write this baseline 3 env

import gymnasium as gym import numpy as np from gymnasium import spaces class CustomEnv(gym.Env):  metadata = {"render_modes": ["human"], "render_fps": 30} def __init__(self, arg1, arg2, ...): super().__init__() # Define action and observation space # They must be gym.spaces objects # Example when using discrete actions: self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) # Example for using image as input (channel-first; channel-last also works): self.observation_space = spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8) def step(self, action): ... return observation, reward, terminated, truncated, info def reset(self, seed=None, options=None): ... return observation, info def render(self): ... def close(self): ...

knowing that the number of varibale in each formula differe


"""
 

#%%

message = [{"role": "user", "content": prompt}]
completion = openai.ChatCompletion.create(
    model=model_engine,
    messages=message,
    temperature = 1,
    # max_tokens=2000,    
    )
response = completion.choices[0].message.content
print(response)
# %%
