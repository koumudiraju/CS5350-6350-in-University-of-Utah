import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("DataSets/train.csv", header=None)
test_data = pd.read_csv("DataSets/test.csv", header=None)
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * gamma ** 2))

def kernel_perceptron(X_train, y_train, X_test, y_test, gamma, max_epochs=10):
    n_samples = X_train.shape[0]
    alphas = np.zeros(n_samples)
    training_errors = []
    test_errors = []
    for epoch in range(max_epochs):
        for i in range(n_samples):
            kernel_sum = np.sum(alphas * y_train * np.array([gaussian_kernel(X_train[j], X_train[i], gamma) for j in range(n_samples)]))
            prediction = np.sign(kernel_sum)
            
            if prediction == 0:  
                prediction = -1
            if prediction != y_train[i]:
                alphas[i] += 1

        y_train_pred = np.sign([np.sum(alphas * y_train * np.array([gaussian_kernel(X_train[j], X_train[i], gamma) for j in range(n_samples)])) for i in range(n_samples)])
        training_errors.append(np.mean(y_train_pred != y_train))

        # Compute test error
        y_test_pred = np.sign([np.sum(alphas * y_train * np.array([gaussian_kernel(X_train[j], X_test[i], gamma) for j in range(n_samples)])) for i in range(X_test.shape[0])])
        test_errors.append(np.mean(y_test_pred != y_test))

    return training_errors[-1], test_errors[-1]

gamma_values = [0.1, 0.5, 1, 5, 100]
for gamma in gamma_values:
    train_error, test_error = kernel_perceptron(X_train, y_train, X_test, y_test, gamma)
    print(f"Gamma: {gamma}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")
