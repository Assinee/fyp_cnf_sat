#%%
import pandas as pd
df = pd.read_csv("/home/assine/fyp/rl_heuristic/testing/results/ppo_5_2_9_10_100_1000000.csv")
print("RL is doing better:",len(df[df["rl_branch_count"]<df["mf_branch_count"]]))
print("RL is doing same as mf:",len(df[df["rl_branch_count"]==df["mf_branch_count"]]))
print("RL is not doing better:",len(df[df["rl_branch_count"]>df["mf_branch_count"]]))
print(df[df["rl_branch_count"]>df["mf_branch_count"]]["rl_result"].value_counts())
print(df[df["rl_branch_count"]>df["mf_branch_count"]]["rl_branch_count"].value_counts())
# %%


