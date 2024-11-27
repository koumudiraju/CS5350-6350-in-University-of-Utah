import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_path = "/content/train.csv"
test_path = "/content/test.csv"

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def svm_sgd(X_train, y_train, X_test, y_test, C, gamma_0, max_epochs):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    objective_values = []
    updates = 0

    for epoch in range(max_epochs):
        indices = np.random.permutation(n_samples)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(n_samples):
            updates += 1
            gamma_t = gamma_0 / (1 + updates) 
            margin = y_train[i] * (np.dot(X_train[i], w) + b)
            if margin < 1:
                w = (1 - gamma_t) * w + gamma_t * C * y_train[i] * X_train[i]
                b += gamma_t * C * y_train[i]
            else:
                w = (1 - gamma_t) * w  # No update to b

            hinge_loss = np.maximum(0, 1 - y_train * (np.dot(X_train, w) + b))
            objective = 0.5 * np.dot(w, w) + C * np.sum(hinge_loss)
            objective_values.append(objective)

    def predict(X):
        return np.sign(np.dot(X, w) + b)

    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)

    train_error = np.mean(y_train_pred != y_train)
    test_error = np.mean(y_test_pred != y_test)

    return w, b, train_error, test_error, objective_values

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_0 = 0.1  # Tune as needed
max_epochs = 100

results = []
for C in C_values:
    w, b, train_error, test_error, objective_values = svm_sgd(X_train, y_train, X_test, y_test, C, gamma_0, max_epochs)
    results.append((C, train_error, test_error))
    print(f"C={C:.3f}, Training Error={train_error:.4f}, Test Error={test_error:.4f}")
