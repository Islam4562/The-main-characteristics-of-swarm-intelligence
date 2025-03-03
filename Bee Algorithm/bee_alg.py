import numpy as np
import pandas as pd

class BeeAlgorithm:
    def __init__(self, num_bees=20, num_elite_sites=5, num_best_sites=10, iterations=100):
        self.num_bees = num_bees
        self.num_elite_sites = num_elite_sites
        self.num_best_sites = num_best_sites
        self.iterations = iterations
    
    def objective_function(self, x):
        return np.sum(x)
    
    def optimize(self, data):
        best_solution = None
        best_score = float('-inf')
        
        for _ in range(self.iterations):
            # Generate random solutions
            solutions = [np.random.choice(data.flatten(), size=len(data), replace=True) for _ in range(self.num_bees)]
            scores = [self.objective_function(sol) for sol in solutions]
            
            # Select the best solutions
            sorted_indices = np.argsort(scores)[::-1]
            best_solutions = [solutions[i] for i in sorted_indices[:self.num_best_sites]]
            best_scores = [scores[i] for i in sorted_indices[:self.num_best_sites]]
            
            if best_scores[0] > best_score:
                best_solution = best_solutions[0]
                best_score = best_scores[0]
        
        return best_solution, best_score

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("ba_data.csv")
    data_values = df.iloc[:, :-1].values  # Use only features

    # Run Bee Algorithm
    ba = BeeAlgorithm()
    best_solution, best_score = ba.optimize(data_values)
    print("Best found solution:", best_solution)
    print("Objective function value:", best_score)