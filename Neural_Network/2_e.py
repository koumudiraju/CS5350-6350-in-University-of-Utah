import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Custom neural network structure
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_type):
        super(NeuralNetworkModel, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        if activation_type == 'tanh':
            nn.init.xavier_uniform_(self.hidden_layer.weight)  # Xavier Initialization
            self.activation = nn.Tanh()
        elif activation_type == 'relu':
            nn.init.kaiming_uniform_(self.hidden_layer.weight, nonlinearity='relu')  # He Initialization
            self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

# Loss calculation function
def compute_loss(model, loss_function, X, y):
    with torch.no_grad():
        predictions = model(X)
        return loss_function(predictions, y).item()

# Training the neural network
def train_network(X_train, y_train, hidden_size, activation_type, lr, epochs):
    input_size = X_train.shape[1]
    output_size = 1  # Assuming binary classification
    model = NeuralNetworkModel(input_size, hidden_size, output_size, activation_type)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = loss_function(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model

# Load and preprocess the dataset
train_dataset = pd.read_csv("/content/train.csv", header=None)
test_dataset = pd.read_csv("/content/test.csv", header=None)

X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values
X_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, -1].values

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameters
depth_options = [3, 5, 9]
width_options = [5, 10, 25, 50, 100]
learning_rate = 0.001
num_epochs = 100

# Train and evaluate the neural network
results_summary = []

for activation_function in ['tanh', 'relu']:
    for hidden_width in width_options:
        for depth in depth_options:
            print(f"Training with Activation: {activation_function}, Width: {hidden_width}, Depth: {depth}")

            # Train the model
            trained_model = train_network(X_train, y_train, hidden_width, activation_function, learning_rate, num_epochs)

            # Compute errors
            train_loss = compute_loss(trained_model, nn.MSELoss(), torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
            test_loss = compute_loss(trained_model, nn.MSELoss(), torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32).view(-1, 1))

            results_summary.append({
                'activation': activation_function,
                'width': hidden_width,
                'depth': depth,
                'train_loss': train_loss,
                'test_loss': test_loss
            })

            print(f"Training Loss: {train_loss:.4f}, Testing Loss: {test_loss:.4f}")
