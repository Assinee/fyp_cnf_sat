import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, formula, max_solution=10, max_conflict=8, alpha=0.1):
        super().__init__()
        self.original_formula = list(formula)
        self.variables = self.get_all_variables(formula)
        self.num_variables = len(self.variables)

        self.action_space = spaces.Discrete(
            self.num_variables * 2  # Each variable can be true or false
        )
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_variables * 2,), dtype=np.float32
        )
        self.step_count = 0
        self.alpha = alpha
        self.max_solution = max_solution
        self.max_conflict = max_conflict
        self.reset()

    def get_all_variables(self, formula):
        return sorted(list(set(abs(num) for sublist in formula for num in sublist)))

    def step(self, action):
        self.step_count += 1
        reward = -self.alpha
        variable_index = action // 2
        variable_value = (action % 2) * 2 - 1  # Translates action to -1 or 1

        if self.assignment[variable_index] != 0:
            return self._get_obs(), -100, True, False, {"message": "Variable already assigned"}

        self.assignment[variable_index] = variable_value
        reward, done = self._evaluate_assignment()

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, return_info=False):
        super().reset(seed=seed)  # Only if using a newer gym version that supports seeding
        self.formula = self.original_formula.copy()
        return self._get_obs(),{}

    def _get_obs(self):
        var_counts = np.zeros(self.num_variables * 2)
        for clause in self.formula:
            for var in clause:
                index = abs(var) - 1
                var_counts[index * 2 + (var > 0)] += 1
        return var_counts

    def _evaluate_assignment(self):
        simplified_formula = []
        num_clauses_before = len(self.formula)
        conflict_found = False

        for clause in self.formula:
            new_clause = [var for var in clause if not self.is_satisfied(var)]
            if len(new_clause) == 0 and any(self.is_satisfied(var, strictly=True) for var in clause):
                conflict_found = True
            elif len(new_clause) > 0:
                simplified_formula.append(new_clause)

        self.formula = simplified_formula
        num_clauses_after = len(self.formula)
        clauses_removed = num_clauses_before - num_clauses_after

        reward = clauses_removed * self.alpha
        if conflict_found:
            reward -= self.max_conflict
            return reward, True
        if num_clauses_after == 0:
            reward += self.max_solution
            return reward, True

        return reward, False

    def is_satisfied(self, var, strictly=False):
        index = abs(var) - 1
        if strictly:
            return self.assignment[index] == np.sign(var)
        else:
            return self.assignment[index] != 0 and self.assignment[index] == np.sign(var)

    def render(self, mode="human"):
        print(f"Remaining formula: {self.formula}")

    def close(self):
        pass
