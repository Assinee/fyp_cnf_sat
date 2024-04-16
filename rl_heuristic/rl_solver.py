import time
import gzip
import numpy as np
from stable_baselines3 import PPO
from sat_env import SatEnv
import gzip
from stable_baselines3.common.env_util import make_vec_env


def read_cnf_file(filename):
    formula = []
    opener = gzip.open if filename.endswith('.gz') else open

    with opener(filename, 'rt') as file:
        for line in file:
            if not (line.startswith('c') or line.startswith('p')):
                clause = []
                for token in line.split():
                    if token != '0' and token.lstrip('-').isdigit():
                        clause.append(int(token))
                if clause:
                    formula.append(clause)

    return formula


def get_rl_variable(formula,assigned_variables):    
    assigned_values = np.zeros(20, dtype=int)
    for i in assigned_variables:
        assigned_values[abs(i)-1] = np.sign(i)
    print(assigned_values)
    env = SatEnv(formula, assigned_values)
    vec_env = make_vec_env(lambda: env, n_envs=1)
    obs = vec_env.reset()

    model = PPO.load("final_model")
    action, _states = model.predict(obs, deterministic=True)
    new_obs, reward, done, info = vec_env.step(action) 
    variable_value=get_action(assigned_values,info[0]["assigned_variables"])    
    env.render()
    return variable_value

def get_action(assigned_values,list_output):
    for i in range(len(list_output)):
        if list_output[i] != assigned_values[i]:
            return list_output[i]*(i+1)
    return 0

def solve(formula, original_formula=[], assigned_variables=[], branch_count=0):
    if assigned_variables == []:
        new_formula = formula
        original_formula = formula
    else:
        variable = assigned_variables[-1]
        new_formula = [clause for clause in formula if variable not in clause]
        new_formula = [[x for x in sublist if x != -variable] for sublist in new_formula]
    if new_formula == []:
       return (assigned_variables, branch_count)
    if any(len(clause) == 0 for clause in new_formula):
        return (None, branch_count)
    new_variable = get_rl_variable(original_formula,assigned_variables)
    assigned_variables.append(new_variable)
    print(new_variable)
    assigned_variables_new= assigned_variables[:]
    result, branch_count = solve(new_formula, original_formula,assigned_variables_new, branch_count + 1)
    if result is not None:
        return (result, branch_count)    
    var = assigned_variables.pop()
    assigned_variables.append(-new_variable)
    assigned_variables_new= assigned_variables[:]
    return solve(new_formula,original_formula, assigned_variables_new, branch_count + 1)



# Example usage:
# formula = [[1, -2, 3], [-1, 3], [-1, 2, 3], [1, -2]]
# formula = [[1, 2], [1, -2], [-1, 2], [-1, -2]]

formula=read_cnf_file('/home/assine/fyp/dataset_fyp/uf20-01.cnf')

start_time = time.time()
result, branch_count = solve(formula)
elapsed_time = time.time() - start_time
print("Time taken:", elapsed_time, "seconds")

if result is not None:
    print("Satisfiable. Assignment:", result)
    print("Numbre of variable assigned",len(result))
    print(f"Number of branches searched: {branch_count}")
else:
    print("Unsatisfiable.")
