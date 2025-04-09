import random
import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions
GRID_SIZE = 20  
NUM_ANTS = 10  # Number of ants
NUM_FOOD = 3   # Number of food sources
PHEROMONE_DECAY = 0.1  # Pheromone decay rate

# Initialize the grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))  # Pheromone field
food_positions = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(NUM_FOOD)]
nest_position = (GRID_SIZE // 2, GRID_SIZE // 2)  # Nest at the center

# Ant class
class Ant:
    def __init__(self):
        self.position = nest_position
        self.has_food = False

    def move(self):
        """Ant chooses a movement based on pheromones or randomness"""
        x, y = self.position
        neighbors = [(x+dx, y+dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= x+dx < GRID_SIZE and 0 <= y+dy < GRID_SIZE]

        if self.has_food:
            # If the ant has food, it moves towards the nest
            self.position = min(neighbors, key=lambda pos: distance(pos, nest_position))
            grid[x, y] += 1  # Leave pheromone
        else:
            # Choose direction: either random or based on pheromone
            if random.random() < 0.7:  # 70% chance to follow pheromone
                self.position = max(neighbors, key=lambda pos: grid[pos[0], pos[1]])
            else:
                self.position = random.choice(neighbors)

        # Check if the ant found food
        if not self.has_food and self.position in food_positions:
            self.has_food = True  # Ant found food
        elif self.has_food and self.position == nest_position:
            self.has_food = False  # Ant delivered food home

# Manhattan distance function
def distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Create ants
ants = [Ant() for _ in range(NUM_ANTS)]

# Simulation loop
steps = 100
for _ in range(steps):
    for ant in ants:
        ant.move()
    grid *= (1 - PHEROMONE_DECAY)  # Pheromones decay over time

# Visualization of results
plt.imshow(grid, cmap='inferno', origin='upper')
plt.scatter(*zip(*food_positions), marker='o', color='green', label="Food")
plt.scatter(nest_position[1], nest_position[0], marker='s', color='blue', label="Nest")
plt.colorbar(label="Pheromone Level")
plt.legend()
plt.title("Ant Colony Simulation (Pheromone Map)")
plt.show()
