#%%
import pandas as pd
df = pd.read_csv("/home/assine/fyp/rl_heuristic/testing/results/ppo_3_1_9_10_100000.csv")
print("ppo_3_1_9_10_100000")
print("RL is doing better:",len(df[df["rl_branch_count"]<df["mf_branch_count"]]))
print("RL is doing same as mf:",len(df[df["rl_branch_count"]==df["mf_branch_count"]]))
print("RL is not doing better:",len(df[df["rl_branch_count"]>df["mf_branch_count"]]))
print(df[df["rl_branch_count"]>df["mf_branch_count"]]["rl_result"].value_counts())

# %%
