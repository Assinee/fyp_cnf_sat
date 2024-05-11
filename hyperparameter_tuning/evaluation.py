#%%
import pandas as pd
df = pd.read_csv("fyp/hyperparameter_tuning/results/ppo_3var_alpha_5_sol_10_conf_9_pen_-200_restart_True_8")
print("RL is doing better:",len(df[df["rl_branch_count"]<df["mf_branch_count"]]))
print("RL is doing same as mf:", len(df[(df["rl_branch_count"] == df["mf_branch_count"]) & (df["rl_result"] != "skipped")]))
print("RL is not doing better:", len(df[(df["rl_branch_count"] > df["mf_branch_count"]) & (df["rl_result"] != "skipped")]))
# %%