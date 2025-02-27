import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define metro network as a graph
metro = nx.Graph()
stations = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
edges = [("A", "B", 3), ("B", "C", 2), ("C", "D", 4), ("D", "E", 1), ("E", "F", 6),
         ("A", "G", 4), ("G", "H", 5), ("H", "I", 2), ("I", "J", 3), ("J", "F", 4),
         ("B", "H", 3), ("C", "I", 5), ("D", "J", 2)]

for edge in edges:
    metro.add_edge(edge[0], edge[1], weight=edge[2], pheromone=1.0)

# ACO parameters
num_ants = 10
num_iterations = 100
evaporation_rate = 0.3
alpha = 1  # Influence of pheromone
beta = 2   # Influence of path weight

# Function to find the best route for an ant
def find_route(start, end):
    current = start
    path = [current]
    visited = set(path)
    while current != end:
        neighbors = [n for n in metro.neighbors(current) if n not in visited]
        if not neighbors:
            return None  # Dead end
        
        # Compute probabilities based on pheromone and weight
        probabilities = []
        for neighbor in neighbors:
            pheromone = metro[current][neighbor]['pheromone'] ** alpha
            weight = (1.0 / metro[current][neighbor]['weight']) ** beta
            probabilities.append(pheromone * weight)
        
        probabilities = np.array(probabilities) / sum(probabilities)
        next_station = np.random.choice(neighbors, p=probabilities)
        path.append(next_station)
        visited.add(next_station)
        current = next_station
    return path

# Function to update pheromones
def update_pheromones(routes):
    for edge in metro.edges():
        metro[edge[0]][edge[1]]['pheromone'] *= (1 - evaporation_rate)  # Evaporation
    
    for route in routes:
        if route:
            for i in range(len(route) - 1):
                metro[route[i]][route[i+1]]['pheromone'] += 1.0 / len(route)  # Deposit pheromone

# Run ACO optimization
best_route = None
best_length = float('inf')
route_history = []

for _ in range(num_iterations):
    routes = [find_route("A", "F") for _ in range(num_ants)]
    update_pheromones(routes)
    
    for route in routes:
        if route:
            length = sum(metro[route[i]][route[i+1]]['weight'] for i in range(len(route) - 1))
            if length < best_length:
                best_length = length
                best_route = route
    route_history.append(best_route)

print("Best route found:", best_route)
print("Best route length:", best_length)

# Animation setup
fig, ax = plt.subplots()
positions = nx.spring_layout(metro)

def update(num):
    ax.clear()
    nx.draw(metro, positions, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, ax=ax)
    if num < len(route_history) and route_history[num]:
        edges = [(route_history[num][i], route_history[num][i+1]) for i in range(len(route_history[num]) - 1)]
        nx.draw_networkx_edges(metro, positions, edgelist=edges, edge_color='red', width=2, ax=ax)

ani = animation.FuncAnimation(fig, update, frames=len(route_history), interval=200)
plt.show()
