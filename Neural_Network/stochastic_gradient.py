import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
train_data = pd.read_csv("/content/train.csv", header=None)
test_data = pd.read_csv("/content/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set hyperparameters
hidden_layer_sizes = [5, 10, 25, 50, 100]
initial_learning_rate = 0.5
decay_factor = 0.001
epochs = 100

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

# Weight initialization
def initialize_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    weights = {
        'hidden': np.random.normal(0, 1, (input_dim, hidden_dim)),
        'output': np.random.normal(0, 1, (hidden_dim, output_dim))
    }
    return weights

# Learning rate schedule
def learning_rate_schedule(gamma_0, decay, t):
    return gamma_0 / (1 + (gamma_0 / decay) * t)

# Forward pass
def forward_pass(X, weights):
    hidden_layer_input = np.dot(X, weights['hidden'])
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights['output'])
    output = sigmoid(output_layer_input)

    return hidden_layer_output, output

# Backpropagation
def backpropagate(X, y, predicted, hidden_output, weights, learning_rate):
    output_error = y - predicted
    output_adjustment = output_error * sigmoid_grad(predicted)

    hidden_error = np.dot(output_adjustment, weights['output'].T)
    hidden_adjustment = hidden_error * sigmoid_grad(hidden_output)

    # Update weights
    weights['output'] += np.outer(hidden_output, output_adjustment) * learning_rate
    weights['hidden'] += np.outer(X, hidden_adjustment) * learning_rate

# Loss computation
def calculate_loss(X, y, weights):
    _, predicted = forward_pass(X, weights)
    loss = np.mean((y - predicted) ** 2)
    return loss

# Training function
def train_nn(X_train, y_train, hidden_dim, output_dim, gamma_0, decay, max_epochs):
    input_dim = X_train.shape[1]
    weights = initialize_weights(input_dim, hidden_dim, output_dim)
    loss_history = []

    for epoch in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        for t, (X, y) in enumerate(zip(X_train, y_train)):
            learning_rate = learning_rate_schedule(gamma_0, decay, epoch * len(X_train) + t)

            hidden_output, predicted = forward_pass(X, weights)
            backpropagate(X, y, predicted, hidden_output, weights, learning_rate)

            if t % 100 == 0:
                loss = calculate_loss(X_train, y_train, weights)
                loss_history.append(loss)

    return weights, loss_history

# Testing function
def evaluate_nn(X_test, y_test, weights):
    _, predicted = forward_pass(X_test, weights)
    predictions = np.round(predicted)
    error_rate = np.mean(predictions != y_test)
    return error_rate

# Store results for different hidden layer sizes
results = []

for hidden_dim in hidden_layer_sizes:
    print(f"\nEvaluating hidden layer size: {hidden_dim}")

    trained_weights, loss_history = train_nn(X_train, y_train, hidden_dim, 1, initial_learning_rate, decay_factor, epochs)

    train_error = evaluate_nn(X_train, y_train, trained_weights)
    test_error = evaluate_nn(X_test, y_test, trained_weights)

    results.append({
        'hidden_layer_size': hidden_dim,
        'training_error': train_error,
        'testing_error': test_error,
        'loss_history': loss_history
    })

    print(f"Training Error: {train_error * 100:.2f}%, Test Error: {test_error * 100:.2f}%")

