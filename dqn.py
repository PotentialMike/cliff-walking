import numpy as np
import tensorflow as tf
import pygame
import lib.agent as agent

from collections import defaultdict
from tqdm import tqdm
from keras.utils.np_utils import to_categorical


### Set config parameters
learning_rate = 0.1
initial_exploration = 1.0
n_episodes = 4
exploration_decay = initial_exploration / (n_episodes / 2)  # reduce the exploration over time
final_exploration = 0.1


### Train the model
# Initialize agent
agent = agent.CliffWalkingAgent(
    learning_rate = learning_rate,
    initial_exploration = initial_exploration,
    exploration_decay = exploration_decay,
    final_exploration = final_exploration
)

# Train model

for episode in tqdm(range(n_episodes)):
    state, info = agent.reset_env()

    # one-hot encode state
    state_encoded = to_categorical(state, agent.env.observation_space.n).shape
    
    done = False
    while not done:
        # use agent policy to pick an action, based on observation (state)
        action = agent.get_action(state_encoded)

        # enact the policy-selected action
        next_state, reward, terminated, truncated, info = agent.env.step(action)
        next_state_encoded = to_categorical(next_state, agent.env.observation_space.n).shape

        # educate the policy with the result
        agent.update(state_encoded, action, reward/100, terminated, truncated, next_state_encoded)

        # update if the environment is done and the current obs
        done = terminated or truncated
        state = next_state

    agent.decay_exploration()
agent.close_env()

# save off model



### Final gameplay loop
print(f"-- FINAL GAMEPLAY --")
agent.init_env(headless=False)
state, info = agent.reset_env()

done = False
step = 100

# while not done:
while step:
    # one-hot encode state
    state_array = to_categorical(state, agent.env.observation_space.n).shape

    # use agent policy to pick an action, based on observation (state)
    action = agent.get_action(state_array)

    # enact the policy-selected action
    next_state, reward, terminated, truncated, info = agent.env.step(action)
    next_state_array = to_categorical(next_state, agent.env.observation_space.n).shape

    # update if the environment is done and the current obs
    step -= 1
    done = terminated or truncated or (step == 0)
    state = next_state

agent.close_env()
