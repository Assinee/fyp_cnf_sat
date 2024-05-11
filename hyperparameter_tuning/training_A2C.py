from cnf_sat_env import SatEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Create training and evaluation environments
env = SatEnv()
train_env = make_vec_env(lambda: env, n_envs=1)
eval_env = make_vec_env(lambda: SatEnv(), n_envs=1)  # Separ for evaluation

# Define model
model = A2C("MlpPolicy", train_env, verbose=1)

# Callbacks
# Checkpoint callback to save model every 1,000,000 steps
checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='/home/assine/fyp/rl_heuristic/hyperparameter_tuning/agents/agents_a2c_5_4_9_10_100_1000000',
                                         name_prefix='rl_agent')
# Evaluation callback to evaluate the model and save the best one
eval_callback = EvalCallback(eval_env, best_model_save_path='/home/assine/fyp/rl_heuristic/hyperparameter_tuning/agents/agents_a2c_5_4_9_10_100_1000000/best_model/',
                             log_path='/home/assine/fyp/rl_heuristic/hyperparameter_tuning/logs/logs_a2c_5_4_9_10_100_1000000/', eval_freq=5000,
                             deterministic=True, render=False)

# Model training with callback
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])

# Save the best model explicitly under a final model name
best_model = A2C.load("/home/assine/fyp/rl_heuristic/hyperparameter_tuning/agents/agents_a2c_5_4_9_10_100_1000000/best_model/best_model")
best_model.save("/home/assine/fyp/rl_heuristic/hyperparameter_tuning/final_model/final_model_a2c_5_4_9_10_100_1000000")
