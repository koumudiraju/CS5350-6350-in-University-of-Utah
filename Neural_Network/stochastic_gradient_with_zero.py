import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sigmoid activation function and its derivative
def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(output):
    return output * (1 - output)

# Initialize weights with zeros
def initialize_weights_zeros(input_size, hidden_size, output_size):
    return {
        'hidden_layer': np.zeros((input_size, hidden_size)),
        'output_layer': np.zeros((hidden_size, output_size))
    }

# Learning rate adjustment schedule
def adjust_learning_rate(initial_lr, decay_factor, iteration):
    return initial_lr / (1 + (initial_lr / decay_factor) * iteration)

# Forward propagation through the layers
def forward_propagation(inputs, weights):
    hidden_input = np.dot(inputs, weights['hidden_layer'])
    hidden_output = activation_sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights['output_layer'])
    final_output = activation_sigmoid(output_input)

    return hidden_output, final_output

# Backpropagation to adjust weights
def back_propagation(inputs, targets, predictions, hidden_output, weights, learning_rate):
    output_error = targets - predictions
    output_adjustment = output_error * sigmoid_gradient(predictions)

    hidden_error = np.dot(output_adjustment, weights['output_layer'].T)
    hidden_adjustment = hidden_error * sigmoid_gradient(hidden_output)

    weights['output_layer'] += np.outer(hidden_output, output_adjustment) * learning_rate
    weights['hidden_layer'] += np.outer(inputs, hidden_adjustment) * learning_rate

# Compute the mean squared error loss for diagnostics
def compute_mean_squared_loss(inputs, targets, weights):
    _, predictions = forward_propagation(inputs, weights)
    return np.mean((targets - predictions) ** 2)

# Train the neural network with zero weights
def train_network_with_zero_weights(inputs_train, targets_train, hidden_size, output_size, initial_lr, decay_factor, num_epochs):
    input_size = inputs_train.shape[1]
    weights = initialize_weights_zeros(input_size, hidden_size, output_size)
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(len(inputs_train))
        inputs_train = inputs_train[indices]
        targets_train = targets_train[indices]

        for step, (inputs, targets) in enumerate(zip(inputs_train, targets_train)):
            learning_rate = adjust_learning_rate(initial_lr, decay_factor, epoch * len(inputs_train) + step)

            hidden_output, predictions = forward_propagation(inputs, weights)
            back_propagation(inputs, targets, predictions, hidden_output, weights, learning_rate)

            if step % 100 == 0:
                loss = compute_mean_squared_loss(inputs_train, targets_train, weights)
                loss_history.append(loss)

    return weights, loss_history

# Evaluate the network's performance
def evaluate_model(inputs, targets, weights):
    _, predictions = forward_propagation(inputs, weights)
    predictions_binary = np.round(predictions)
    return 1 - np.mean(predictions_binary == targets)  # Return error rate

# Load and normalize the dataset
train_data = pd.read_csv("/content/train.csv", header=None)
test_data = pd.read_csv("/content/test.csv", header=None)

inputs_train = train_data.iloc[:, :-1].values
targets_train = train_data.iloc[:, -1].values
inputs_test = test_data.iloc[:, :-1].values
targets_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
inputs_train = scaler.fit_transform(inputs_train)
inputs_test = scaler.transform(inputs_test)

# Hyperparameters
hidden_layer_sizes = [5, 10, 25, 50, 100]
initial_learning_rate = 0.5
decay_factor = 0.001
total_epochs = 100

training_results = []

for hidden_size in hidden_layer_sizes:
    print(f"\nEvaluating hidden layer size:  {hidden_size} (Zero Weights Initialization)")

    # Train the neural network
    trained_weights, loss_curve = train_network_with_zero_weights(inputs_train, targets_train, hidden_size, 1, initial_learning_rate, decay_factor, total_epochs)

    # Evaluate the trained model
    train_error = evaluate_model(inputs_train, targets_train, trained_weights)
    test_error = evaluate_model(inputs_test, targets_test, trained_weights)

    training_results.append({
        'hidden_layer_size': hidden_size,
        'train_error': train_error,
        'test_error': test_error,
        'loss_history': loss_curve
    })

    print(f"Training Error: {train_error * 100:.2f}%, Test Error: {test_error * 100:.2f}%")
