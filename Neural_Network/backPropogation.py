import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Initialize weights for the neural network
def initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim):
    np.random.seed(42)
    weights = {
        'layer1': np.random.randn(input_dim, hidden_dim1) * 0.1,
        'layer2': np.random.randn(hidden_dim1, hidden_dim2) * 0.1,
        'output': np.random.randn(hidden_dim2, output_dim) * 0.1
    }
    return weights

# Perform forward propagation
def forward_pass(inputs, weights):
    l1_input = np.dot(inputs, weights['layer1'])
    l1_output = relu(l1_input)

    l2_input = np.dot(l1_output, weights['layer2'])
    l2_output = relu(l2_input)

    final_input = np.dot(l2_output, weights['output'])
    final_output = sigmoid(final_input)

    return l1_output, l2_output, final_output

# Backpropagation to compute gradients and update weights
def backpropagation(inputs, target, outputs, l2_output, l1_output, weights, lr, reg_lambda):
    output_error = target - outputs
    output_delta = output_error * sigmoid_derivative(outputs)

    l2_error = np.dot(output_delta, weights['output'].T)
    l2_delta = l2_error * relu_derivative(l2_output)

    l1_error = np.dot(l2_delta, weights['layer2'].T)
    l1_delta = l1_error * relu_derivative(l1_output)

    # Update weights with L2 regularization
    weights['output'] += lr * (np.outer(l2_output, output_delta) - reg_lambda * weights['output'])
    weights['layer2'] += lr * (np.outer(l1_output, l2_delta) - reg_lambda * weights['layer2'])
    weights['layer1'] += lr * (np.outer(inputs, l1_delta) - reg_lambda * weights['layer1'])

# Train the neural network
def train_nn(X_train, y_train, hidden_dim1, hidden_dim2, output_dim, lr, epochs, reg_lambda):
    input_dim = X_train.shape[1]
    weights = initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim)
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            inputs = X_train[i]
            target = y_train[i]

            l1_output, l2_output, outputs = forward_pass(inputs, weights)
            loss = 0.5 * (target - outputs) ** 2
            epoch_loss += loss

            backpropagation(inputs, target, outputs, l2_output, l1_output, weights, lr, reg_lambda)

        loss_history.append(epoch_loss / len(X_train))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(X_train)}")

    return weights, loss_history

# Test the neural network
def evaluate_nn(X_test, y_test, weights):
    predictions = []
    for i in range(len(X_test)):
        _, _, output = forward_pass(X_test[i], weights)
        predictions.append(int(output > 0.5))

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return accuracy, precision, recall, f1

# Load the dataset
train_data = pd.read_csv("/content/train.csv", header=None)
test_data = pd.read_csv("/content/test.csv", header=None)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network architecture
hidden_dim1 = 5
hidden_dim2 = 5
output_dim = 1
learning_rate = 0.01
epochs = 200
reg_lambda = 0.01  # Regularization strength

# Train the neural network
trained_weights, loss_history = train_nn(X_train, y_train, hidden_dim1, hidden_dim2, output_dim, learning_rate, epochs, reg_lambda)

# Evaluate the neural network
accuracy, precision, recall, f1 = evaluate_nn(X_test, y_test, trained_weights)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
