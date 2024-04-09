from training import read_cnf_file
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO  # or any other algorithm from Stable Baselines3
from sat_env import SatEnv

# Load the best model
best_model = PPO.load("/home/assine/fyp/rl_heuristic/sat_solver_model.zip")
formula=read_cnf_file('/home/assine/fyp/dataset_fyp/uf20-01.cnf')

# Create a new environment for inference
inference_env = SatEnv(formula)
# check_env(inference_env)

# Reset the environment
obs = inference_env.reset()

# Run inference
while True:
    action, _state = best_model.predict(obs, deterministic=True)
    obs, reward, done, info = inference_env.step(action)
    print(info)
    if done:
        print("Satisfiable. Assignment:", info["assignment"])
        break
    elif truncated:
        print("Truncated episode.")
        break

