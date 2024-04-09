import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sat_env import SatEnv

# Assume SatEnv and read_cnf_file are defined as shown previously
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

# Use your custom evaluate function
env = SatEnv(read_cnf_file("/home/assine/fyp/dataset_fyp/uf20-01.cnf"))
vec_env = make_vec_env(lambda: env, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)
obs = vec_env.reset()

action, _states = model.predict(obs, deterministic=True)

new_obs, reward, done, info = vec_env.step(action)
env.render()
print(f"Action: {action}, Reward: {reward}, Done: {done}")

