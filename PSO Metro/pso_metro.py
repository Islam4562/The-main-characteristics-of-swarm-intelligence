import numpy as np
import networkx as nx
import random

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
Q = 1.0    # Pheromone constant

def calculate_probabilities(current, neighbors):
    pheromones = np.array([metro[current][n]['pheromone'] ** alpha for n in neighbors])
    weights = np.array([(1.0 / metro[current][n]['weight']) ** beta for n in neighbors])
    probabilities = pheromones * weights
    return probabilities / probabilities.sum()

def find_route(start, end):
    current = start
    path = [current]
    visited = set(path)
    while current != end:
        neighbors = [n for n in metro.neighbors(current) if n not in visited]
        if not neighbors:
            return None  # Dead end
        probabilities = calculate_probabilities(current, neighbors)
        next_station = np.random.choice(neighbors, p=probabilities)
        path.append(next_station)
        visited.add(next_station)
        current = next_station
    return path

def evaporate_pheromones():
    for edge in metro.edges():
        metro[edge[0]][edge[1]]['pheromone'] *= (1 - evaporation_rate)

def deposit_pheromones(routes):
    for route in routes:
        if route:
            length = sum(metro[route[i]][route[i+1]]['weight'] for i in range(len(route) - 1))
            pheromone_deposit = Q / length
            for i in range(len(route) - 1):
                metro[route[i]][route[i+1]]['pheromone'] += pheromone_deposit

def optimize_route():
    best_route = None
    best_length = float('inf')
    for _ in range(num_iterations):
        routes = [find_route("A", "F") for _ in range(num_ants)]
        evaporate_pheromones()
        deposit_pheromones(routes)
        for route in routes:
            if route:
                length = sum(metro[route[i]][route[i+1]]['weight'] for i in range(len(route) - 1))
                if length < best_length:
                    best_length = length
                    best_route = route
    return best_route, best_length

best_route, best_length = optimize_route()
print("Best route found:", best_route)
print("Best route length:", best_length)
