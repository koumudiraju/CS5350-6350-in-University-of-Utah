import numpy as np
import pandas as pd
from sklearn.utils import shuffle

train_path = "/content/train.csv"
test_path = "/content/test.csv"

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values


y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)
def svm_sgd(X_train, y_train, X_test, y_test, C, gamma_0, a, max_epochs, schedule_type="schedule_1"):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    updates = 0

    for epoch in range(max_epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=epoch)

        for i in range(n_samples):
            updates += 1
            if schedule_type == "schedule_1":  # γ_t = γ_0 / (1 + t)
                gamma_t = gamma_0 / (1 + updates)
            elif schedule_type == "schedule_2":  # γ_t = γ_0 / (1 + γ_0 / a * t)
                gamma_t = gamma_0 / (1 + (gamma_0 / a) * updates)
            else:
                raise ValueError("Invalid schedule type")

            margin = y_train[i] * (np.dot(X_train[i], w) + b)
            if margin < 1:
                w = (1 - gamma_t) * w + gamma_t * C * y_train[i] * X_train[i]
                b += gamma_t * C * y_train[i]
            else:
                w = (1 - gamma_t) * w  # No update to b

    def predict(X):
        return np.sign(np.dot(X, w) + b)

    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)

    train_error = np.mean(y_train_pred != y_train)
    test_error = np.mean(y_test_pred != y_test)

    return w, b, train_error, test_error

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_0 = 0.1
a = 0.01
max_epochs = 100

for C in C_values:
    print(f"Results for C = {C:.3f}:")

 
    w1, b1, train_error1, test_error1 = svm_sgd(X_train, y_train, X_test, y_test, C, gamma_0, a, max_epochs, "schedule_1")
    w2, b2, train_error2, test_error2 = svm_sgd(X_train, y_train, X_test, y_test, C, gamma_0, a, max_epochs, "schedule_2")

    weight_diff = np.linalg.norm(w1 - w2)
    bias_diff = abs(b1 - b2)
    train_error_diff = abs(train_error1 - train_error2)
    test_error_diff = abs(test_error1 - test_error2)

    print(f"Weight Difference: {weight_diff:.4f}")
    print(f"Bias Difference: {bias_diff:.4f}")
    print(f"Training Error Difference: {train_error_diff:.4f}")
    print(f"Test Error Difference: {test_error_diff:.4f}")
    print("=" * 50)
