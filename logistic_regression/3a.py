import numpy as np
import pandas as pd

# Load and preprocess the dataset
train_dataset = pd.read_csv("/content/train.csv", header=None)
test_dataset = pd.read_csv("/content/test.csv", header=None)

X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values
X_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, -1].values

# Set constants
epochs = 100
prior_variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

# Define sigmoid function
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

# Define learning rate adjustment schedule
def adjust_learning_rate(initial_rate, decay_factor, step):
    return initial_rate / (1 + (initial_rate / decay_factor) * step)

# Objective function including Gaussian prior
def compute_objective(y_actual, y_predicted, weights, prior_variance):
    log_likelihood = -np.sum(y_actual * np.log(y_predicted) + (1 - y_actual) * np.log(1 - y_predicted))
    regularization = 0.5 / prior_variance * np.sum(weights ** 2)
    return log_likelihood + regularization

# Compute gradient of the objective function
def compute_gradient(y_actual, y_predicted, X, weights, prior_variance):
    return -np.dot(X.T, y_actual - y_predicted) + (1 / prior_variance) * weights

# Train logistic regression model with Gaussian prior
def train_logistic_model(X_train, y_train, prior_variance, initial_rate, decay_factor, num_epochs):
    num_features = X_train.shape[1]
    weights = np.zeros(num_features + 1)  # Initialize weights with bias term

    for epoch in range(num_epochs):
        # Shuffle training data at the start of each epoch
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for idx in range(len(X_shuffled)):
            X_sample = np.append(X_shuffled[idx], 1)  # Add bias term
            y_sample = y_shuffled[idx]

            # Calculate dynamic learning rate
            learning_rate = adjust_learning_rate(initial_rate, decay_factor, epoch * len(X_shuffled) + idx)

            # Perform forward pass
            y_predicted = sigmoid_function(np.dot(weights, X_sample))

            # Compute loss and gradient
            loss = compute_objective(y_sample, y_predicted, weights, prior_variance)
            gradient = compute_gradient(y_sample, y_predicted, X_sample, weights, prior_variance)

            # Update weights
            weights -= learning_rate * gradient

    return weights

# Evaluate the model's performance
def evaluate_model(weights, X, y):
    X_with_bias = np.column_stack([X, np.ones(len(X))])  # Add bias term
    predictions = sigmoid_function(np.dot(X_with_bias, weights))
    classified_predictions = (predictions > 0.5).astype(int)
    error_rate = np.mean(classified_predictions != y)
    return error_rate

# Train and evaluate logistic regression for different variances
for variance in prior_variances:
    print(f"\nTraining with Prior Variance: {variance}")

    # Train the model
    trained_weights = train_logistic_model(X_train, y_train, variance, initial_rate=0.1, decay_factor=0.01, num_epochs=epochs)

    # Evaluate on the training set
    training_error = evaluate_model(trained_weights, X_train, y_train)
    print(f"Training Error: {training_error:.4f}")

    # Evaluate on the test set
    testing_error = evaluate_model(trained_weights, X_test, y_test)
    print(f"Testing Error: {testing_error:.4f}")
