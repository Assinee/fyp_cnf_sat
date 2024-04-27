from collections import Counter
import time
import torch
import pandas as pd
import ast
import gzip


df = pd.read_csv('/home/assine/fyp/dataset_generated/cnf_sat_15_variables.csv')
df["rl_result"]=""
df["rl_branch_count"]=""
df["mf_result"]=""
df["mf_branch_count"]=""
for i in range(df.shape[0]):

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


    def get_most_frequent_variable(formula):
        flat_list_abs = [abs(num) for sublist in formula for num in sublist]
        flat_list = [num for sublist in formula for num in sublist]
        counts = Counter(flat_list_abs)
        variable = max(counts, key=counts.get)
        positive_count = flat_list.count(variable)
        negative_count = flat_list.count(-variable)
        if positive_count >= negative_count:
            return variable
        else:
            return -variable   
        
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
        return clean_formula
    
    def solve(formula, assigned_variables=[], branch_count=0):
        if check_formula(formula)=="unsatisfiable":
            print(formula)
            return (None, branch_count)
        else :
            formula = check_formula(formula)
        if assigned_variables == []:
            new_formula = formula
        else:
            variable = assigned_variables[-1]
            print(variable)
            new_formula = [clause for clause in formula if variable not in clause]
            new_formula = [[x for x in sublist if x != -variable] for sublist in new_formula]
        if new_formula == []:
            return (assigned_variables, branch_count)
        if any(len(clause) == 0 for clause in new_formula):
            return (None, branch_count)
        new_variable = get_most_frequent_variable(new_formula)    
        assigned_variables.append(new_variable)
        assigned_variables_new= assigned_variables[:]
        result, branch_count = solve(new_formula, assigned_variables_new, branch_count + 1)
        if result is not None:
            return (result, branch_count)    
        var = assigned_variables.pop()
        assigned_variables.append(-new_variable)
        assigned_variables_new= assigned_variables[:]
        return solve(new_formula, assigned_variables_new, branch_count + 1)



    # Example usage:
    # formula = [[1, -2, 3], [-1, 3], [-1, 2, 3], [1, -2]]
    # formula = [[1, 2], [1, -2], [-1, 2], [-1, -2]]
    formula= ast.literal_eval(df.iloc[i]["formula"])

    # formula=read_cnf_file('/home/assine/fyp/dataset_fyp/uf20-01.cnf')

    start_time = time.time()
    result, branch_count = solve(formula)
    elapsed_time = time.time() - start_time
    df.loc[i, "mf_result"] = str(result)
    df.loc[i, "mf_branch_count"]= str(branch_count)
    print("Time taken:", elapsed_time, "seconds")

    if result is not None:
        print("Satisfiable. Assignment:", result)
        print("Numbre of variable assigned",len(result))
        print(f"Number of branches searched: {branch_count}")
    else:
        print("Unsatisfiable.")
        df.loc[i, "mf_result"] = "unsatisfiable."
    df.to_csv('/home/assine/fyp/dataset_generated/cnf_sat_15_variables.csv',index=False)


