import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Define the Ant Agent
class Ant(Agent):
    """An agent representing an ant in the simulation."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.position = (model.grid.width // 2, model.grid.height // 2)  # Start at the center
        self.has_food = False
    
    def move(self):
        """Ants move based on pheromone levels or randomly."""
        x, y = self.position
        neighbors = self.model.grid.get_neighborhood(self.position, moore=True, include_center=False)
        
        if self.has_food:
            # If the ant has food, it will move towards the nest
            self.position = min(neighbors, key=lambda pos: self.distance(pos, (self.model.grid.width // 2, self.model.grid.height // 2)))
        else:
            # If searching for food, either follow pheromones or move randomly
            if random.random() < 0.7:  # 70% chance to follow pheromones
                self.position = max(neighbors, key=lambda pos: self.model.pheromone_grid[pos[0], pos[1]])
            else:
                self.position = random.choice(neighbors)
                
        self.model.grid.move_agent(self, self.position)
        
        # Check if the ant has found food
        if not self.has_food and self.position in self.model.food_positions:
            self.has_food = True  # Ant found food
        elif self.has_food and self.position == (self.model.grid.width // 2, self.model.grid.height // 2):
            self.has_food = False  # Ant delivered food to the nest

    def distance(self, pos1, pos2):
        """Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Define the Ant Colony Model
class AntColonyModel(Model):
    """A model with multiple ants and a food source."""
    
    def __init__(self, width, height, num_ants, num_food):
        self.num_agents = num_ants
        self.num_food = num_food
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.pheromone_grid = np.zeros((width, height))  # Pheromone grid
        self.food_positions = [(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(num_food)]
        
        # Create agents (ants)
        for i in range(self.num_agents):
            a = Ant(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, a.position)
        
        # Data collector to gather data over time
        self.datacollector = DataCollector(
            agent_reporters={"Position": "position"}
        )
    
    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Pheromone decay
        self.pheromone_grid *= (1 - 0.1)
        
        # After each step, ants leave pheromones on their current location
        for agent in self.schedule.agents:
            x, y = agent.position
            if agent.has_food:  # Only leave pheromones if the ant has food
                self.pheromone_grid[x, y] += 1

# Run the Model
model = AntColonyModel(20, 20, num_ants=10, num_food=3)

# Run for 100 steps
for _ in range(100):
    model.step()

# Visualize the Pheromone Grid
plt.imshow(model.pheromone_grid, cmap='inferno', origin='upper')
plt.scatter(*zip(*model.food_positions), marker='o', color='green', label="Food")
plt.scatter(model.grid.width // 2, model.grid.height // 2, marker='s', color='blue', label="Nest")
plt.colorbar(label="Pheromone Level")
plt.legend()
plt.title("Ant Colony Simulation with Adaptation (Mesa)")
plt.show()
