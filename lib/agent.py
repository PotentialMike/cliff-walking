import gymnasium as gym
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense


from collections import defaultdict

class CliffWalkingAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_exploration: float,
        exploration_decay: float,
        final_exploration: float,
        discount_factor: float = 0.95
    ):
        self.learning_rate = learning_rate
        self.exploration = initial_exploration
        self.exploration_decay = exploration_decay
        self.final_exploration = final_exploration
        self.discount_factor = discount_factor

        # self.training_error = []
        self.init_env()
        obs_space_size = self.env.observation_space.n
        action_space_size = self.env.action_space.n

        self.init_model(obs_space_size, action_space_size)

    
    # Environment functions
    def init_env(self, headless=True):
        if headless:
            self.env = gym.make("CliffWalking-v0")
        else:
            self.env = gym.make("CliffWalking-v0",render_mode='human')


    def reset_env(self):
        return self.env.reset()


    def close_env(self):
        self.env.close()


    # Model functions
    def init_model(self, obs_space_size, action_space_size):
        width = 8
        
        self.model = Sequential()
        self.model.add(Dense(obs_space_size, input_dim=1, activation='relu'))
        self.model.add(Dense(width,activation='relu'))
        self.model.add(Dense(width,activation='relu'))
        self.model.add(Dense(action_space_size, activation='softmax'))

        # compile the model
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # print(self.model.summary())

    
    def get_action(self, state):
        # with probability exploration return a random action to explore the environment
        if np.random.random() < self.exploration:
            return self.env.action_space.sample()

        # with probability (1 - exploration), act greedily (exploit)
        else:
            return int(np.argmax(self.model.predict(state), axis=1))
            
            
    def update(self, state, action, reward, terminated, truncated, next_state):
        future_q_value = (not terminated) * np.max(self.model.predict(next_state))
        temporal_difference = reward + self.discount_factor * future_q_value - self.model.predict(state)
        self.model.fit(state, temporal_difference, verbose=0)


    # def save_model(self):
    

    # def load_model(self):

    
    def update(
        self,
        state: int,
        action: int,
        reward: int,
        terminated: bool,
        truncated: bool,
        next_state: int
    ):
        self.model

    def decay_exploration(self):
        self.exploration = max(self.final_exploration, self.exploration - self.exploration_decay)


class CustomQLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred-y_true))
    rmse = tf.math.sqrt(mse)
    return rmse / tf.reduce_mean(tf.square(y_true)) - 1
