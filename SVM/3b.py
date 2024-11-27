import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

def compute_rbf_kernel(X, gamma):
    sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * np.dot(X, X.T)
    return np.exp(-sq_dists / (2 * gamma**2))

def train_dual_svm_rbf(X, y, C, gamma):
    K = compute_rbf_kernel(X, gamma)
    n_samples = X.shape[0]

    def dual_objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K * np.outer(y, y), alpha)) - np.sum(alpha)

    constraints = {"type": "eq", "fun": lambda alpha: np.dot(alpha, y)}
    bounds = [(0, C) for _ in range(n_samples)]
    initial_alpha = np.zeros(n_samples)

    result = minimize(dual_objective, initial_alpha, method="SLSQP", bounds=bounds, constraints=constraints)
    alpha_opt = result.x
    sv_indices = alpha_opt > 1e-5
    b = np.mean(y[sv_indices] - np.dot(K[sv_indices], alpha_opt * y))
    return alpha_opt, b, sv_indices

def predict_svm_rbf(X, X_sv, y_sv, alpha_sv, b, gamma):
    sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X_sv**2, axis=1)[None, :] - 2 * np.dot(X, X_sv.T)
    kernel_values = np.exp(-sq_dists / (2 * gamma**2))
    return np.sign(np.dot(kernel_values, alpha_sv * y_sv) + b)

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100 / 873, 500 / 873, 700 / 873]

best_combination = None
lowest_test_error = float("inf")

for gamma in gamma_values:
    for C in C_values:
        alpha, b, sv_indices = train_dual_svm_rbf(X_train, y_train, C, gamma)
        X_sv = X_train[sv_indices]
        y_sv = y_train[sv_indices]
        alpha_sv = alpha[sv_indices]

        train_predictions = predict_svm_rbf(X_train, X_sv, y_sv, alpha_sv, b, gamma)
        train_error = np.mean(train_predictions != y_train)

        test_predictions = predict_svm_rbf(X_test, X_sv, y_sv, alpha_sv, b, gamma)
        test_error = np.mean(test_predictions != y_test)

        print(f"Gamma: {gamma:.3f}, C: {C:.3f}")
        print(f"Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")
        print("=" * 50)

        if test_error < lowest_test_error:
            lowest_test_error = test_error
            best_combination = {"gamma": gamma, "C": C}

print(f"Best Combination: Gamma = {best_combination['gamma']}, C = {best_combination['C']}")
print(f"Lowest Test Error: {lowest_test_error:.4f}")
