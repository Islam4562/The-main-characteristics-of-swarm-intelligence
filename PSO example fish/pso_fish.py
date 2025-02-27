import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data representing fish characteristics
# Features: weight, length, width, height
# Target: fish species classification (0: Small Fish, 1: Medium Fish, 2: Large Fish)
def generate_fish_data(num_samples=500):
    X = np.random.rand(num_samples, 4) * [10, 50, 20, 15]  # Random fish dimensions
    y = np.random.randint(0, 3, num_samples)  # Random species labels
    return X, y

# Load fish dataset
X, y = generate_fish_data()
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define a simple feedforward neural network for fish classification
class FishClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(FishClassifier, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)  # Input layer
        self.fc2 = nn.Linear(hidden_size, 3)  # Output layer for 3 classes
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Function to evaluate the neural network
def evaluate_model(params):
    hidden_size = int(params[0])
    learning_rate = params[1]
    model = FishClassifier(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for _ in range(100):  # Training for 100 epochs
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
    return test_loss

# Define Particle class for PSO optimization
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1])
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.best_position = self.position.copy()
        self.best_value = float('inf')
    
    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(2)
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best_position - self.position))
    
    def move(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[:, 0], bounds[:, 1])

# Define PSO class for optimizing the neural network hyperparameters
class PSO:
    def __init__(self, num_particles, bounds, max_iter=30):
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position
        self.global_best_value = float('inf')
        self.max_iter = max_iter
        self.bounds = bounds

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                value = evaluate_model(particle.position)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = particle.position.copy()
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = particle.position.copy()
                
                particle.update_velocity(self.global_best_position)
                particle.move(self.bounds)
        return self.global_best_position, self.global_best_value

# Define search space: number of neurons (10-100), learning rate (0.0001-0.1)
bounds = np.array([[10, 100], [0.0001, 0.1]])

# Run PSO to optimize FishClassifier hyperparameters
pso = PSO(num_particles=20, bounds=bounds, max_iter=30)
opt_params, opt_value = pso.optimize()

print("Optimal parameters:", opt_params)
print("Optimal loss:", opt_value)
