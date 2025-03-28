import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from matplotlib import pyplot as plt

import gymnasium as gym
import gym_race
"""
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
"""

VERSION_NAME = 'DQN_v01'  # the name for our model

env = gym.make("Pyrace-v1").unwrapped  # skip the TimeLimiting and OrderEnforcing default wrappers
obs, _ = env.reset()
print('observation shape', type(obs), obs)
print('env', type(env))
if not os.path.exists(f'models_{VERSION_NAME}'):
    os.makedirs(f'models_{VERSION_NAME}')

# Hyperparameters
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 1000
report_every = 50

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Use the environment's observation_space to get the correct input dimension
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# Function to select an action with the epsilon greedy policy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Select a random action
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state)
        return torch.argmax(q_values).item()  # return the action with the highest q-value
    
# Function to optimize the model with experience replay
def optimize_model():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(np.array(reward_batch))
    next_state_batch = torch.FloatTensor(np.array(next_state_batch))
    done_batch = torch.FloatTensor(done_batch)

    # Compute Q-values
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Use the target network to compute the target Q-values
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
    
    target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)
    
    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
rewards_for_episode = []
steps_done = 0

for episode in range(episodes):
    obs, _ = env.reset()
    state = obs
    total_reward = 0
    done = False

    while not done:
        # Select an action with epsilon-greedy policy
        action = select_action(state, epsilon)
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = next_obs

        # Save the transition in memory
        memory.append((state, action, reward, next_state, float(done)))

        state = next_state
        total_reward += reward

        optimize_model()

        # Update the target network periodically
        if steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        steps_done += 1

    # Decay epsilon after each episode
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    rewards_for_episode.append(total_reward)

    if episode % report_every == 0:
        print(f"Episode {episode} - Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

print("Training Complete")
plt.plot(rewards_for_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Performance')
plt.show()
