import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

# Define the Q-network (neural network for Q-learning)
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(input_size, 128)
        # Hidden layer
        self.fc2 = nn.Linear(128, 64)
        # Output layer (4 possible actions: up, down, left, right)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the environment (grid world for the agent to move in)
class Environment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        # Randomly place food and agent in the grid
        self.food_position = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]
        self.agent_position = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]

    def reset(self):
        # Reset agent to a random position
        self.agent_position = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        return np.array(self.agent_position)

    def step(self, action):
        """Move the agent based on the action and return reward"""
        if action == 0:  # Up
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 1:  # Down
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)
        elif action == 2:  # Left
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 3:  # Right
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)

        # Check if the agent has reached the food
        reward = -1  # Default reward (negative for each move)
        done = False
        if self.agent_position == self.food_position:
            reward = 10  # Reward for reaching the food
            done = True

        return np.array(self.agent_position), reward, done

# Hyperparameters for training
grid_size = 5
input_size = 2  # Agent's position (x, y)
output_size = 4  # 4 possible actions (up, down, left, right)
learning_rate = 0.001
gamma = 0.99  # Discount factor for future rewards
epsilon = 0.1  # Epsilon for epsilon-greedy policy
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 32
num_episodes = 1000

# Initialize the model, optimizer, and loss function
model = QNetwork(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Initialize the environment
env = Environment()

# List to store the total reward for each episode (for plotting)
reward_list = []

# Main training loop
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0

    while not done:
        # Choose action (epsilon-greedy policy)
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3])  # Random action
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()  # Action with the highest Q-value

        # Take the chosen action in the environment
        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

        # Update the Q-value using the Bellman equation
        with torch.no_grad():
            q_values_next = model(next_state)
            target = reward + gamma * torch.max(q_values_next).item() * (1 - int(done))

        # Get the current Q-value for the chosen action
        q_values = model(state)
        loss = loss_fn(q_values[action], torch.tensor(target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    # Decay epsilon after each episode (to reduce exploration over time)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Append the total reward for the episode to the list
    reward_list.append(total_reward)

    # Print the total reward for every 100th episode
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Visualization

# 1. Plot the total reward per episode
plt.figure(figsize=(12, 6))
plt.plot(reward_list)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Agent\'s Total Reward per Episode')
plt.show()

# 2. Visualize the final state of the agent and food
plt.figure(figsize=(6, 6))
plt.imshow(np.zeros((grid_size, grid_size)), cmap='gray', origin='upper', extent=(0, grid_size, 0, grid_size))
plt.scatter(*env.food_position, color='red', label='Food', s=100)
plt.scatter(*env.agent_position, color='blue', label='Agent', s=100)
plt.title(f"Final Agent and Food State")
plt.legend()
plt.grid(True)
plt.show()
