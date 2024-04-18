from env import TicTacToeEnv
from stable_baselines3 import PPO

# Load the model; ensure you've trained and saved it first
model = PPO.load("ppo_tictactoe")
import numpy as np

def play_against_ai(env, model):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        # Current player is human (assuming player 2 is the AI)
        if env.current_player == 1:
            action = int(input("Your move (0-8): "))
            while env.state[action] != 0:
                action = int(input("Invalid move, try again (0-8): "))
        else:
            action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, _, info= env.step(action)
        if done:
            env.render()
            if reward == 1:
                print("Player 1 wins!" if env.current_player == 2 else "Player 2 wins!")
            elif reward == -1:
                print("Player 2 wins!" if env.current_player == 1 else "Player 1 wins!")
            else:
                print("It's a draw!")
                
env = TicTacToeEnv()  # Make sure the environment starts with the human player
play_against_ai(env, model)
