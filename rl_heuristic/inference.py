import gymnasium as gym
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

# assigned_variables = [-5]
# assigned_values = np.zeros(20, dtype=int)
# for i in assigned_variables:
#     assigned_values[abs(i)-1] = np.sign(i)
# print(assigned_values)
env = SatEnv(read_cnf_file("/home/assine/fyp/dataset_fyp/uf20-02.cnf"))
vec_env = make_vec_env(lambda: env, n_envs=1)
obs = vec_env.reset()

model = PPO.load("final_model")
action, _states = model.predict(obs, deterministic=True)

new_obs, reward, done, info = vec_env.step(action)
env.render()
print(f"Action: {((action//2)+1)*((action % 2) * 2 - 1)},Info: {info} , reward:{reward}")
