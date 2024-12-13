import numpy as np
import pandas as pd

# Load and preprocess the dataset
train_dataset = pd.read_csv("/content/train.csv", header=None)
test_dataset = pd.read_csv("/content/test.csv", header=None)

X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values
X_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, -1].values

# Constants
num_epochs = 100
variance_values = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

# Sigmoid activation function
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

# Learning rate adjustment
def dynamic_learning_rate(initial_lr, decay_factor, iteration):
    return initial_lr / (1 + (initial_lr / decay_factor) * iteration)

# Logistic likelihood function
def compute_likelihood(y_actual, y_predicted):
    return -np.sum(y_actual * np.log(y_predicted) + (1 - y_actual) * np.log(1 - y_predicted))

# Compute gradient of the loss function
def compute_gradient(y_actual, y_predicted, features):
    return -np.dot(features.T, y_actual - y_predicted)

# Train logistic regression model (ML Estimation)
def train_logistic_model(X_train, y_train, initial_lr, decay_factor, epochs):
    num_features = X_train.shape[1]
    weights = np.zeros(num_features + 1)  # Initialize weights with bias term

    for epoch in range(epochs):
        # Shuffle training data at the start of each epoch
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(len(X_shuffled)):
            # Add bias term to the feature vector
            X_sample = np.append(X_shuffled[i], 1)
            y_sample = y_shuffled[i]

            # Update learning rate dynamically
            learning_rate = dynamic_learning_rate(initial_lr, decay_factor, epoch * len(X_shuffled) + i)

            # Forward pass: predict and calculate loss
            y_predicted = sigmoid_function(np.dot(weights, X_sample))
            loss = compute_likelihood(y_sample, y_predicted)

            # Backward pass: calculate gradient and update weights
            gradients = compute_gradient(y_sample, y_predicted, X_sample)
            weights -= learning_rate * gradients

    return weights

# Evaluate model performance
def evaluate_model(weights, X, y):
    # Add bias term to the dataset
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    predictions = sigmoid_function(np.dot(X_with_bias, weights))
    classified_predictions = (predictions > 0.5).astype(int)
    error_rate = np.mean(classified_predictions != y)
    return error_rate

# Train and evaluate logistic regression for different variances
for variance in variance_values:
    print(f"\nTraining ML Estimation with Variance Setting: {variance}")

    # Train the logistic regression model
    trained_weights = train_logistic_model(X_train, y_train, initial_lr=0.1, decay_factor=0.01, epochs=num_epochs)

    # Evaluate on training data
    training_error = evaluate_model(trained_weights, X_train, y_train)
    print(f"Training Error: {training_error:.4f}")

    # Evaluate on test data
    testing_error = evaluate_model(trained_weights, X_test, y_test)
    print(f"Test Error: {testing_error:.4f}")
