# =============================
# Reinforcement Learning Taxi Agent
# =============================

import gymnasium as gym
import numpy as np
import random

# Create environment
env = gym.make("Taxi-v3")

# State and action space
state_size = env.observation_space.n
action_size = env.action_space.n

print("States:", state_size)
print("Actions:", action_size)

# Create Q-table
q_table = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 1.0     # exploration rate

epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 10000

# Training loop
for episode in range(episodes):

    state, _ = env.reset()
    done = False

    while not done:

        # Explore vs exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Q-learning formula
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        q_table[state, action] = new_value

        state = next_state

    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training finished!")

np.save("q_table.npy", q_table)