import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("DataSets/train.csv", header=None)
train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

def compute_rbf_kernel(X, gamma):
    sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * np.dot(X, X.T)
    return np.exp(-sq_dists / (2 * gamma ** 2))

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
    return alpha_opt, sv_indices

gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]
C_values = [100 / 873, 500 / 873, 700 / 873]

support_vector_counts = {}
overlap_counts = {}
for C in C_values:
    print(f"\nC = {C:.3f}:")
    prev_sv_indices = None  # To store support vectors from the previous gamma
    for gamma in gamma_values:
        alpha, sv_indices = train_dual_svm_rbf(X_train, y_train, C, gamma)
        num_support_vectors = np.sum(sv_indices)
        support_vector_counts[(C, gamma)] = num_support_vectors

        print(f"Gamma: {gamma:.3f}, Number of Support Vectors: {num_support_vectors}")

        if C == 500 / 873 and prev_sv_indices is not None:
            overlap = np.sum(prev_sv_indices & sv_indices)  # Count overlapping support vectors
            overlap_counts[(gamma_prev, gamma)] = overlap
            print(f"Overlap with Gamma = {gamma_prev:.3f}: {overlap}")

        # Update for the next iteration
        prev_sv_indices = sv_indices
        gamma_prev = gamma

if 500 / 873 in C_values:
    print("\nOverlap Counts for C = 500/873:")
    for (gamma_prev, gamma), overlap in overlap_counts.items():
        print(f"Gamma = {gamma_prev:.3f} -> Gamma = {gamma:.3f}: Overlap = {overlap}")

