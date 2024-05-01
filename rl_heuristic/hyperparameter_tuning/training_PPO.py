from itertools import product
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from cnf_sat_env import SatEnv

# Define the ranges or specific values for each parameter
alphas = [3, 4, 5]  # Example values for alpha
max_solutions = [10]  # Example values for max_solution
max_conflicts = [9]  # Example values for max_conflict
unvalid_action_penalties = [-100, -200, -300]  # Example penalties
with_restarts = [False]
max_stepss = [6, 8]  # Only relevant if with_restart is True

def train_and_evaluate(env):
    """Train and evaluate the model with the given environment settings"""
    train_env = make_vec_env(lambda: env, n_envs=1)
    eval_env = make_vec_env(lambda: env, n_envs=1)  # Separate for evaluation

    model = PPO("MlpPolicy", train_env, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path=f"./agents/agents_ppo_3var_alpha_{alpha}_sol_{max_solution}_conf_{max_conflict}_pen_{unvalid_action_penalty}_restart_{with_restart}",
        name_prefix="rl_agent",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./agents/agents_ppo_3var_alpha_{alpha}_sol_{max_solution}_conf_{max_conflict}_pen_{unvalid_action_penalty}_restart_{with_restart}/best_model/",
        log_path=f"./logs/logs_ppo_3var_alpha_{alpha}_sol_{max_solution}_conf_{max_conflict}_pen_{unvalid_action_penalty}_restart_{with_restart}/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])

    # Optionally save the best model explicitly under a final model name
    best_model = PPO.load(
        f"./agents/agents_ppo_3var_alpha_{alpha}_sol_{max_solution}_conf_{max_conflict}_pen_{unvalid_action_penalty}_restart_{with_restart}/best_model/best_model"
    )
    best_model.save(
        f"final_model/final_model_ppo_3var_alpha_{alpha}_sol_{max_solution}_conf_{max_conflict}_pen_{unvalid_action_penalty}_restart_{with_restart}"
    )

# Iterate over all combinations of parameters
for alpha, max_solution, max_conflict, unvalid_action_penalty, with_restart in product(
    alphas, max_solutions, max_conflicts, unvalid_action_penalties, with_restarts
):

    if with_restart:
        for max_steps in max_stepss:
            max_step_penalty = unvalid_action_penalty
            # Initialize the environment with the current set of parameters
            env = SatEnv(
                alpha,
                max_solution,
                max_conflict,
                unvalid_action_penalty,
                with_restart,
                max_steps,
                max_step_penalty,
            )
            train_and_evaluate(env)
    else:
        # Set max_steps and max_step_penalty to zero when with_restart is False
        env = SatEnv(
            alpha,
            max_solution,
            max_conflict,
            unvalid_action_penalty,
            with_restart,
            0,
            0,
        )
        train_and_evaluate(env)



# Note: Remember to adjust paths and parameters as per your directory structure and requirements.
