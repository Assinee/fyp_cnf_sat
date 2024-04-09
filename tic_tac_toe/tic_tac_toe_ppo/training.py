import gym
from stable_baselines3 import PPO

env = TicTacToeEnv()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

