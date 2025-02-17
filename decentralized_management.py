import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Parameters for the model
GRID_SIZE = 20
NUM_ANTS = 10
NUM_FOOD = 3
PHEROMONE_DECAY = 0.1
NUM_STEPS = 100

class Ant(Agent):
    """An agent representing an ant in the colony."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.position = (GRID_SIZE // 2, GRID_SIZE // 2)  # Starting at the center (nest)
        self.has_food = False
    
    def move(self):
        """The ant moves either randomly or follows pheromones."""
        x, y = self.position
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE]
        
        # Ants return to the nest when they have food
        if self.has_food:
            self.position = min(neighbors, key=lambda pos: distance(pos, (GRID_SIZE//2, GRID_SIZE//2)))
        else:
            # Ants follow pheromones or move randomly
            if random.random() < 0.7:  # 70% chance to follow pheromones
                self.position = max(neighbors, key=lambda pos: self.model.pheromone_grid[pos[0], pos[1]])
            else:
                self.position = random.choice(neighbors)
        
        # Check for food
        if not self.has_food and self.position in self.model.food_positions:
            self.has_food = True  # Ant found food
        elif self.has_food and self.position == (GRID_SIZE // 2, GRID_SIZE // 2):
            self.has_food = False  # Ant returns food to the nest

    def step(self):
        """This method is called to update the ant's behavior at each step."""
        self.move()
        # Leave pheromone when bringing food to nest
        if self.has_food:
            x, y = self.position
            self.model.pheromone_grid[x, y] += 1  # Add pheromone

class AntColonyModel(Model):
    """A model of an ant colony."""
    def __init__(self, num_ants, num_food, grid_size=GRID_SIZE):
        self.num_ants = num_ants
        self.num_food = num_food
        self.grid_size = grid_size
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.grid_size, self.grid_size, True)
        self.food_positions = [(random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)) for _ in range(self.num_food)]
        
        # Pheromone grid
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))
        
        # Create the ants
        for i in range(self.num_ants):
            a = Ant(i, self)
            self.schedule.add(a)
        
        # Data collector to track statistics
        self.datacollector = DataCollector(
            agent_reporters={"Position": "position", "Has_Food": "has_food"}
        )
    
    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.pheromone_grid *= (1 - PHEROMONE_DECAY)  # Decay the pheromones
        self.schedule.step()

# Function to calculate Manhattan distance between two positions
def distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Run the model
model = AntColonyModel(NUM_ANTS, NUM_FOOD)
for i in range(NUM_STEPS):
    model.step()

# Visualizing the pheromone grid
plt.imshow(model.pheromone_grid, cmap='inferno', origin='upper')
plt.scatter(*zip(*model.food_positions), marker='o', color='green', label="Food")
plt.scatter(GRID_SIZE // 2, GRID_SIZE // 2, marker='s', color='blue', label="Nest")
plt.colorbar(label="Pheromone Level")
plt.legend()
plt.title("Ant Colony Simulation (Decentralized Control)")
plt.show()
