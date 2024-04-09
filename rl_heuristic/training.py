import gym
import numpy as np
from sat_env import SatEnv
from stable_baselines3 import PPO
import os
import gzip
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv


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

# cnf_files_directory = "/home/assine/fyp/dataset_fyp"
# cnf_files = [os.path.join(cnf_files_directory, f) for f in os.listdir(cnf_files_directory) if f.endswith('.cnf')]

# # Creating a list of environments
# envs = [lambda: SatEnv(read_cnf_file(cnf_file)) for cnf_file in cnf_files]  # Lazy initialization of environments

env=SatEnv(read_cnf_file("/home/assine/fyp/dataset_fyp/uf20-01.cnf"))

model = PPO("MlpPolicy", envs, verbose=1)

# Directory to save all models
models_dir = "agents/sat_solver_models"
os.makedirs(models_dir, exist_ok=True)

best_performance = -float('inf')  # Initialize with a very low value
best_model_path = None


def evaluate_agent(model, env, num_episodes=10):
    total_rewards = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
    average_reward = total_rewards / num_episodes
    return average_reward

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

    # Evaluate model
    average_reward = evaluate_agent(model, envs[0]())  # Assuming envs[0]() is a valid environment for evaluation
    print(f"Iteration {iters}, Average Reward: {average_reward}")

    # Save model
    model_path = os.path.join(models_dir, f"sat_solver_model_{TIMESTEPS*iters}.zip")
    model.save(model_path)

    # Update best model if current model is better
    if average_reward > best_performance:
        best_performance = average_reward
        best_model_path = model_path
        print("New best model saved")

# After training, you may want to copy or move the best model to a specific "best" directory
best_models_dir = "agents/best_sat_solver_model"
os.makedirs(best_models_dir, exist_ok=True)
if best_model_path:
    best_model_filename = os.path.basename(best_model_path)
    os.rename(best_model_path, os.path.join(best_models_dir, best_model_filename))
    print(f"Best model moved to {best_models_dir}")
