import sys, os, math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some TkAgg backend memory issues
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race  # this registers the env "Pyrace-v1" from race_env.py and pyrace_2d.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

VERSION_NAME     = 'DQN_v02'         # Model/version name for saving models and plots
REPORT_EPISODES  = 500               # Report (plot/save model) every REPORT_EPISODES episodes
DISPLAY_EPISODES = 100               # Render/display the game every DISPLAY_EPISODES episodes

EPISODES         = 10000             # Total number of episodes to train
MAX_T            = 1000              # Maximum timesteps per episode

# DQN Hyperparameters
LEARNING_RATE    = 0.001
GAMMA            = 0.99
EPSILON_START    = 1.0
EPSILON_MIN      = 0.01
EPSILON_DECAY    = 0.9
BATCH_SIZE       = 64
TARGET_UPDATE_FREQ = 1000
MEMORY_SIZE      = 10000

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

env = gym.make("Pyrace-v1").unwrapped  # skip TimeLimiting and OrderEnforcing wrappers
print("Environment:", type(env))

# Create directory for models if it doesn't exist
if not os.path.exists(f'models_{VERSION_NAME}'):
    os.makedirs(f'models_{VERSION_NAME}')

# Use the environment's observation space for the network input size
input_dim  = env.observation_space.shape[0]
output_dim = env.action_space.n
print(f"Input dimension: {input_dim}, Number of actions: {output_dim}")


policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # target network in evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)
steps_done = 0

def select_action(state, epsilon):
    """Selects an action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        # Convert state to torch tensor (batch size 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

def optimize_model():
    """Sample a minibatch from memory and update the policy network."""
    global steps_done
    if len(memory) < BATCH_SIZE:
        return

    transitions = random.sample(memory, BATCH_SIZE)
    # Unpack the batch
    states, actions, rewards, next_states, dones = zip(*transitions)
    states      = torch.FloatTensor(np.array(states))
    actions     = torch.LongTensor(actions).unsqueeze(1)
    rewards     = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones       = torch.FloatTensor(dones)

    # Current Q-values for the actions taken
    q_values = policy_net(states).gather(1, actions).squeeze(1)
    # Compute the target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q = rewards + GAMMA * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    steps_done += 1
    # Update target network periodically
    if steps_done % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

def simulate(learning=True, episode_start=0):
    """Main simulation loop for training (or playing) the DQN agent."""
    epsilon = EPSILON_START
    rewards_history = []
    max_reward = -1e4  # track maximum reward seen

    for episode in range(episode_start, EPISODES + episode_start):
        obs, _ = env.reset()
        # Ensure state is a numpy float array
        state = np.array(obs, dtype=np.float32)
        total_reward = 0

        #if not learning and hasattr(env, "pyrace"):
        #    env.pyrace.mode = 2

        for t in range(MAX_T):
            action = select_action(state, epsilon if learning else 0)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_obs, dtype=np.float32)
            total_reward += reward

            # Save transition to replay memory
            memory.append((state, action, reward, next_state, float(done)))
            state = next_state

            if learning:
                optimize_model()

            # Render the environment if requested
            if episode % DISPLAY_EPISODES == 0:
                msgs = [
                    'SIMULATE',
                    f'Episode: {episode}',
                    f'Time steps: {t}',
                    f'Reward: {total_reward:.0f}',
                    f'Max Reward: {max_reward:.0f}'
                ]
                if hasattr(env, "set_msgs"):
                    env.set_msgs(msgs)
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        # Decay epsilon (only during learning)
        if learning:
            epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        rewards_history.append(total_reward)

        # Reporting and saving
        if episode % REPORT_EPISODES == 0:
            print(f"Episode {episode} - Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
            plt.figure()
            plt.plot(rewards_history)
            plt.ylabel('Total Reward')
            plt.xlabel('Episode')
            plt.title('DQN Training Performance')
            plot_file = f'models_{VERSION_NAME}/reward_{episode}.png'
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved reward plot to {plot_file}")

            if hasattr(env, "save_memory"):
                mem_file = f'models_{VERSION_NAME}/memory_{episode}.npy'
                env.save_memory(mem_file)
                print(f"Saved memory to {mem_file}")

            # Save the model weights
            model_file = f'models_{VERSION_NAME}/dqn_{episode}.pth'
            torch.save(policy_net.state_dict(), model_file)
            print(f"Saved model to {model_file}")

    print("Training complete.")

def load_and_play(model_episode, learning=False):
    """Load a saved model and then run simulation (training or just play)."""
    model_file = f'models_{VERSION_NAME}/dqn_{model_episode}.pth'
    if os.path.exists(model_file):
        policy_net.load_state_dict(torch.load(model_file))
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Loaded model from {model_file}")
    else:
        print(f"Model file {model_file} not found. Starting from scratch.")

    simulate(learning=learning, episode_start=model_episode)

if __name__ == "__main__":
    # Uncomment one of the following options:
    
    # Option 1: Train the DQN agent from scratch.
    # simulate(learning=True, episode_start=0)
    
    # Option 2: Load a saved model and play (or continue training).
    load_and_play(4000, learning=False)