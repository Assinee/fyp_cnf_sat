from env import TicTacToeEnv
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Assuming the TicTacToeEnv class definition is already available

# Make environment
env = make_vec_env(lambda: TicTacToeEnv(), n_envs=1)

# Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Save the model
model.save("ppo_tictactoe")

# Close the environment
env.close()
