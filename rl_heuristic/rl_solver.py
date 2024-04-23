import time
import gzip
import numpy as np
from stable_baselines3 import A2C
from cnf_sat_env import SatEnv
import gzip
from stable_baselines3.common.env_util import make_vec_env

env = SatEnv()
model = A2C.load("/home/assine/fyp/rl_heuristic/final_model/final_model_a2c_3_1_8_10_1000000.zip")

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

def get_observation(formula ,max_nb_clause):
    nb_variable=3
    observation=[]
    for clause in formula:
        observation_clause=[0]*nb_variable
        for element in clause:
            observation_clause[abs(element)-1]=np.sign(element)
        observation.append(observation_clause)
    zero_clauses = np.zeros((max_nb_clause-len(observation), nb_variable), dtype=np.int8)
    observation = np.vstack((observation, zero_clauses))
    return observation


def get_rl_variable(formula):
    print(formula)      
    observation=get_observation(formula,27)
    env.observation = observation
    action, _states = model.predict(observation, deterministic=True)
    new_observation, reward, done, _, info = env.step(action)
    variable_index=(action//2)+1
    variable_sign= 1 if action % 2 == 0 else -1
    return variable_index*variable_sign

def check_formula(formula):
    clean_formula = set()
    for clause in formula:
        clause_tuple = tuple(clause)  
        if len(clause) == 1:
            literal = clause[0]
            if (-literal,) in clean_formula:  
                return "unsatisfiable"
        else:
            clause_set = set(clause)
            for literal in clause:
                if -literal in clause_set:
                    return "unsatisfiable"
        clean_formula.add(clause_tuple)  
    return [list(clause) for clause in clean_formula]


def solve(formula, assigned_variables=[], branch_count=0):
    if check_formula(formula)=="unsatisfiable":
        return (None, branch_count)
    else :
        formula = check_formula(formula)
    if assigned_variables == []:
        new_formula = formula
    else:
        variable = assigned_variables[-1]
        new_formula = [clause for clause in formula if variable not in clause]
        new_formula = [[x for x in sublist if x != -variable] for sublist in new_formula]
    if new_formula == []:
       return (assigned_variables, branch_count)
    if any(len(clause) == 0 for clause in new_formula):
        return (None, branch_count)
    new_variable = get_rl_variable(new_formula)
    print(new_variable)
    assigned_variables.append(new_variable)
    assigned_variables_new= assigned_variables[:]
    result, branch_count = solve(new_formula,assigned_variables_new, branch_count + 1)
    if result is not None:
        return (result, branch_count)    
    var = assigned_variables.pop()
    assigned_variables.append(-new_variable)
    assigned_variables_new= assigned_variables[:]
    return solve(new_formula, assigned_variables_new, branch_count + 1)



# Example usage:
# formula = [[1, -2, 3], [-1, 3], [-1, 2, 3], [1, -2]]
# formula = [[1, 2], [1, -2], [-1, 2], [-1, -2]]
# formula = [[1,2],[-1,3],[-2,-3],[-1,2],[1,-3]]
# formula = [[-1, 2], [1, -3], [-1, -2], [2, 3]]
# formula= [[1, 2], [-1, -3], [1, -2], [2, 3]]
formula= [[-1, 2, -3], [-1, -2, 3], [1, -3], [1, 2]]
# formula= [[-1, 2], [-1, -3], [1, -2], [2], [1, 2, -3], [-2, 3], [1, 3], [-1, -2], [3]]
# formula=[[-1, 2, 3], [1, -3], [-1, -2], [2], [1, -2], [3], [-2, -3], [1, 3], [-1, 1]]


# formula=read_cnf_file('/home/assine/fyp/dataset_fyp/uf20-01.cnf')

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
