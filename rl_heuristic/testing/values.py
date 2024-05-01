#%%
import pandas as pd
df = pd.read_csv("/home/assine/fyp/rl_heuristic/hyperparameter_tuning/results/ppo_3var_alpha_4_sol_10_conf_9_pen_-200_restart_False")
print("RL is doing better:",len(df[df["rl_branch_count"]<df["mf_branch_count"]]))
print("RL is doing same as mf:",len(df[df["rl_branch_count"]==df["mf_branch_count"]]))
print("RL is not doing better:",len(df[df["rl_branch_count"]>df["mf_branch_count"]]))
print(df[df["rl_branch_count"]>df["mf_branch_count"]]["rl_result"].value_counts())
print(df[df["rl_branch_count"]>df["mf_branch_count"]]["rl_branch_count"].value_counts())
# %%


