import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
file_path = "hyperparameter_data.csv"
df = pd.read_csv(file_path)

# Split data into features and target variable
X = df.drop(columns=['score'])
y = df['score']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter space
param_space = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': np.arange(3, 15),
    'min_samples_split': np.arange(2, 10),
    'min_samples_leaf': np.arange(1, 10)
}

# ACO hyperparameters
num_ants = 10  # Number of ants
num_iterations = 20  # Number of iterations
pheromone = np.ones((num_ants, len(param_space)))  # Pheromone matrix
alpha = 1.0  # Pheromone influence
beta = 2.0  # Heuristic influence
rho = 0.5  # Evaporation rate

# Evaluation function
def evaluate_params(params):
    model = RandomForestRegressor(
        n_estimators=int(params[0]),
        max_depth=int(params[1]),
        min_samples_split=int(params[2]),
        min_samples_leaf=int(params[3]),
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return -mse  # Negative MSE, as ACO seeks to maximize fitness

# ACO algorithm
best_score = float('-inf')
best_params = None

for iteration in range(num_iterations):
    solutions = []
    scores = []
    
    for ant in range(num_ants):
        params = [
            np.random.choice(param_space['n_estimators']),
            np.random.choice(param_space['max_depth']),
            np.random.choice(param_space['min_samples_split']),
            np.random.choice(param_space['min_samples_leaf'])
        ]
        score = evaluate_params(params)
        solutions.append(params)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    # Update pheromone levels
    pheromone *= (1 - rho)
    for i, score in enumerate(scores):
        pheromone[i] += score / abs(best_score)
    
    print(f"Iteration {iteration + 1}/{num_iterations}: Best Score = {best_score}")

# Print best hyperparameters
print("Best Hyperparameters:", best_params)
