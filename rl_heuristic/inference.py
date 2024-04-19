import numpy as np
from stable_baselines3 import PPO
from cnf_sat_env import SatEnv

# Create an environment instance if you don't already have one
env = SatEnv()
model = PPO.load("final_model")


# Set a specific observation manually
# Make sure your observation conforms to the environment's observation space
# custom_observation = np.random.randint(-1, 1, (27, 3), dtype=np.int8)  # Example random observation
# formula = [[1, -2, 3], [-1, 3], [-1, 2, 3], [1, -2]]

initial_clauses = np.array([[1, -1, 1],
                            [-1, 0, 1],
                            [-1, 1, 1],
                             [1, -1, 0]], dtype=np.int8)

# Create 23 clauses of all zeros
zero_clauses = np.zeros((23, 3), dtype=np.int8)

# Concatenate the initial clauses with the zero clauses to form the full observation
custom_observation = np.vstack((initial_clauses, zero_clauses))
print(custom_observation)
env.observation = custom_observation

# Reset the environment's internal state if necessary (not just the otion)

# Reset environment to get initial state
observation = env.reset()

# Loop until the episode is done
current_observation= custom_observation
done = False
while not done:
    # Use your model to predict the action
    # Assume 'model' is your trained PPO model
    action, _states = model.predict(current_observation, deterministic=True)
    env.reset()
    # Apply the action to the environment to get the new state and reward
    new_observation, reward, done, _, info = env.step(action)
    current_observation=new_observation
    # Output the results
    print("Action Taken:", action)
    print("Reward Received:", reward)
    print("New Observation:", new_observation)
    print("Done:", done)
    print("Info:", info)

    # Update the observation with the new state
    observation = new_observation

# Close the environment
env.close()
