from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt

# Ant agent definition
class Ant(Agent):
    """An ant agent that moves around the grid, follows pheromones, and collects food."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.has_food = False

    def move(self):
        """Move the ant to a new position based on pheromone concentration or random."""
        x, y = self.pos
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if self.has_food:
            # Return to nest
            nest = self.model.nest_pos
            self.model.grid.move_agent(self, nest)
        else:
            # Move to a location with the highest pheromone
            next_pos = max(neighbors, key=lambda pos: self.model.grid.get_cell_list_contents([pos]).count("food"))
            self.model.grid.move_agent(self, next_pos)

        # Find food
        if not self.has_food and self.model.grid.get_cell_list_contents([self.pos]) == ["food"]:
            self.has_food = True

    def step(self):
        self.move()

# The model itself
class AntColonyModel(Model):
    """Model representing an ant colony where ants move around to collect food."""
    def __init__(self, width, height, num_ants, num_food):
        self.num_agents = num_ants
        self.num_food = num_food
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Create ants
        for i in range(self.num_agents):
            a = Ant(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Place food sources randomly
        for i in range(self.num_food):
            food_pos = (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height))
            self.grid.place_agent("food", food_pos)

        # Data collection
        self.datacollector = DataCollector(
            agent_reporters={"Food": "has_food"}
        )

        self.nest_pos = (self.grid.width // 2, self.grid.height // 2)

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

# Running the simulation
model = AntColonyModel(width=40, height=40, num_ants=200, num_food=50)
steps = 100
for i in range(steps):
    model.step()

# Visualizing the result
food_positions = [(x, y) for x in range(model.grid.width) for y in range(model.grid.height) if model.grid.get_cell_list_contents([(x, y)]) == ["food"]]

# Plotting ants' movement and food positions
fig, ax = plt.subplots()
ax.set_title("Ant Colony Simulation (Scalable Model)")
ax.set_xlim(0, model.grid.width)
ax.set_ylim(0, model.grid.height)

# Food visualization
food_x, food_y = zip(*food_positions)
ax.scatter(food_x, food_y, c='green', marker='o', label='Food')

# Ants' movement (to be enhanced for visualization)
ants_pos = [ant.pos for ant in model.schedule.agents if isinstance(ant, Ant)]
ants_x, ants_y = zip(*ants_pos)
ax.scatter(ants_x, ants_y, c='red', marker='x', label='Ants')

plt.legend()
plt.show()
