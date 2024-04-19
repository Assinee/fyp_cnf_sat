import os
from cnf_sat_env import SatEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Create training and evaluation environments
env = SatEnv()
train_env = make_vec_env(lambda: env, n_envs=1)
eval_env = make_vec_env(lambda: SatEnv(), n_envs=1)  # Separ for evaluation

# Define model
model = PPO("MlpPolicy", train_env, verbose=1)

# Callbacks
# Checkpoint callback to save model every 1,000,000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./agents/',
                                         name_prefix='rl_agent')
# Evaluation callback to evaluate the model and save the best one
eval_callback = EvalCallback(eval_env, best_model_save_path='./agents/best_model/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)

# Model training with callback
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])

# Save the best model explicitly under a final model name
best_model = PPO.load("./agents/best_model/best_model")
best_model.save("final_model")
