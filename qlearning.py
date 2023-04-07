# pip install gymnasium box2d box2d-kengz pygame tqdm
import gymnasium as gym
import numpy as np
import pygame

from collections import defaultdict
from tqdm import tqdm

import lib.agent as agent


# learning_rate = 0.01
# initial_exploration = 1.0
# n_episodes = 100_000

learning_rate = 0.1
initial_exploration = 1.0
n_episodes = 10_000

exploration_decay = initial_exploration / (n_episodes / 2)  # reduce the exploration over time
final_exploration = 0.1

class CliffWalkingAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_exploration: float,
        exploration_decay: float,
        final_exploration: float,
        discount_factor: float = 0.95,
    ):
        # Initialize Q-values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.exploration = initial_exploration
        self.exploration_decay = exploration_decay
        self.final_exploration = final_exploration

        self.training_error = []
        

    def get_action(self, state):
        # with probability exploration return a random action to explore the environment
        if np.random.random() < self.exploration:
            return env.action_space.sample()

        # with probability (1 - exploration), act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[state]))


    def update(
        self,
        state: int,
        action: int,
        reward: int,
        terminated: bool,
        truncated: bool,
        next_state: int
    ):
        
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[state][action]
        )

        self.q_values[state][action] = (
            self.q_values[state][action] + self.learning_rate * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_exploration(self):
        self.exploration = max(self.final_exploration, self.exploration - exploration_decay)

agent = CliffWalkingAgent(
    learning_rate = learning_rate,
    initial_exploration = initial_exploration,
    exploration_decay = exploration_decay,
    final_exploration = final_exploration
)

env = gym.make("CliffWalking-v0")

for episode in tqdm(range(n_episodes)):
    state, info = env.reset()
    done = False

    while not done:
        # use agent policy to pick an action, based on observation (state)
        action = agent.get_action(state)

        # enact the policy-selected action
        next_state, reward, terminated, truncated, info = env.step(action)

        # educate the policy with the result
        agent.update(state, action, reward, terminated, truncated, next_state)

        # update if the environment is done and the current obs
        done = terminated or truncated
        state = next_state

    agent.decay_exploration()

env.close()

# Final Playthrough
env = gym.make("CliffWalking-v0", render_mode="human")
state, info = env.reset()

pygame.display.set_caption("CliffWalking-v0")

done = False
energy = 25

while not done:
    # use agent policy to pick an action, based on observation (state)
    action = agent.get_action(state)

    # enact the policy-selected action
    next_state, reward, terminated, truncated, info = env.step(action)
    energy += reward

    # update if the environment is done and the current obs
    done = terminated or truncated or (energy == 0)
    state = next_state
  
env.close()