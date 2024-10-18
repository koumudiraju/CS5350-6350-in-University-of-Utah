import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
train_data_path = '/mnt/data/train.csv'
test_data_path = '/mnt/data/test.csv'

# Load train and test data
train_df = pd.read_csv(train_data_path, header=None)
test_df = pd.read_csv(test_data_path, header=None)

# Based on the description, the first 7 columns are features, and the last column is the target (SLUMP)
X_train = train_df.iloc[:, :-1].values  # First 7 columns (features)
y_train = train_df.iloc[:, -1].values   # Last column (target SLUMP)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Normalize the features for better convergence
mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
X_train_normalized = (X_train - mean_X_train) / std_X_train

mean_X_test = np.mean(X_test, axis=0)
std_X_test = np.std(X_test, axis=0)
X_test_normalized = (X_test - mean_X_test) / std_X_test

# Add bias term (column of ones) to X_train and X_test
X_train_normalized = np.hstack([np.ones((X_train_normalized.shape[0], 1)), X_train_normalized])
X_test_normalized = np.hstack([np.ones((X_test_normalized.shape[0], 1)), X_test_normalized])

# Initialize weights
weights = np.zeros(X_train_normalized.shape[1])

# Set learning rate and tolerance
learning_rate = 0.01
max_iterations = 50000  # Maximum number of SGD updates

# Function to compute the cost (mean squared error)
def compute_cost(X, y, weights):
    predictions = X.dot(weights)
    cost = np.mean((predictions - y) ** 2) / 2
    return cost

# Stochastic Gradient Descent algorithm
def stochastic_gradient_descent(X, y, weights, learning_rate, max_iterations):
    cost_history = []
    
    for iteration in range(max_iterations):
        # Randomly shuffle the training data
        idx = np.random.randint(0, X.shape[0])
        X_sample = X[idx, :].reshape(1, -1)
        y_sample = y[idx]
        
        # Compute predictions and gradient for the sampled data point
        prediction = X_sample.dot(weights)
        error = prediction - y_sample
        gradient = X_sample.T.dot(error)
        
        # Update weights
        weights = weights - learning_rate * gradient.flatten()
        
        # Compute cost over the entire dataset for convergence tracking
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
        
        # Reduce learning rate over time for stability
        if iteration % 10000 == 0:
            learning_rate *= 0.9  # Decrease learning rate every 10,000 updates

    return weights, cost_history

# Tune learning rate for SGD
learning_rate = 0.01
final_weights, cost_history = stochastic_gradient_descent(X_train_normalized, y_train, weights, learning_rate, max_iterations)

# Plot cost function over iterations
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function over Iterations (SGD)')
plt.show()

# Compute the cost on the test set using the final weights
test_cost = compute_cost(X_test_normalized, y_test, final_weights)
print(f"Final learned weights: {final_weights}")
print(f"Cost on test data: {test_cost}")
print(f"Learning rate used: {learning_rate}")
