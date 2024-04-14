import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from sat_env import SatEnv
import gzip

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

assigned_variables = [15, -5, -12, -7]
assigned_values = np.zeros(20, dtype=int)
for i in assigned_variables:
    assigned_values[abs(i)-1] = np.sign(i)

env = SatEnv(read_cnf_file("/home/assine/fyp/dataset_fyp/uf20-01.cnf"), assigned_values)
vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./agents/',
                                         name_prefix='rl_agent')

model.learn(total_timesteps=100000, callback=checkpoint_callback)
model.save("final_model")

