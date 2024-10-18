import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data_path = 'DataSets/Bank/train.csv'
test_data_path = 'DataSets/Bank/test.csv'
train_df = pd.read_csv(train_data_path, header=None)
test_df = pd.read_csv(test_data_path, header=None)

X_train = train_df.iloc[:, :-1].values  # First 7 columns (features)
y_train = train_df.iloc[:, -1].values   # Last column (target SLUMP)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
X_train_normalized = (X_train - mean_X_train) / std_X_train

mean_X_test = np.mean(X_test, axis=0)
std_X_test = np.std(X_test, axis=0)
X_test_normalized = (X_test - mean_X_test) / std_X_test

X_train_normalized = np.hstack([np.ones((X_train_normalized.shape[0], 1)), X_train_normalized])
X_test_normalized = np.hstack([np.ones((X_test_normalized.shape[0], 1)), X_test_normalized])

weights = np.zeros(X_train_normalized.shape[1])

learning_rate = 1.0
tolerance = 1e-6
max_iterations = 10000

def compute_cost(X, y, weights):
    predictions = X.dot(weights)
    cost = np.mean((predictions - y) ** 2) / 2
    return cost

def batch_gradient_descent(X, y, weights, learning_rate, tolerance, max_iterations):
    cost_history = []
    weight_diff_history = []
    
    for iteration in range(max_iterations):
        predictions = X.dot(weights)
        errors = predictions - y
        gradient = X.T.dot(errors) / len(y)
        
        new_weights = weights - learning_rate * gradient
        weight_diff = np.linalg.norm(new_weights - weights)
        cost = compute_cost(X, y, new_weights)
        
        weights = new_weights
        cost_history.append(cost)
        weight_diff_history.append(weight_diff)
        if weight_diff < tolerance:
            print(f"Convergence reached after {iteration} iterations.")
            break
    
    return weights, cost_history, weight_diff_history

learning_rates = [1.0, 0.5, 0.25, 0.125, 0.0625]
final_weights = None
cost_history = None
learning_rate_used = None

for lr in learning_rates:
    print(f"Trying learning rate: {lr}")
    weights = np.zeros(X_train_normalized.shape[1])  # Reset weights
    final_weights, cost_history, weight_diff_history = batch_gradient_descent(
        X_train_normalized, y_train, weights, lr, tolerance, max_iterations
    )
    
    if len(cost_history) > 1 and weight_diff_history[-1] < tolerance:
        learning_rate_used = lr
        break

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title(f'Cost Function over Iterations (Learning rate: {learning_rate_used})')
plt.show()

test_cost = compute_cost(X_test_normalized, y_test, final_weights)
print(f"Final learned weights: {final_weights}")
print(f"Cost on test data: {test_cost}")
print(f"Learning rate used: {learning_rate_used}")
