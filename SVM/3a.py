import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

train_data = pd.read_csv("DataSets/train.csv", header=None)
test_data = pd.read_csv("DataSets/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

def svm_primal(X, y, C, initial_lr, decay_rate, epochs):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features) 
    bias = 0  
    updates = 0

    for epoch in range(epochs):
        X, y = shuffle(X, y, random_state=epoch)  # Shuffle data every epoch
        for i in range(n_samples):
            updates += 1
            lr = initial_lr / (1 + (initial_lr / decay_rate) * updates)  # Learning rate schedule
            margin = y[i] * (np.dot(X[i], weights) + bias)
            if margin < 1:
                weights = (1 - lr) * weights + lr * C * y[i] * X[i]
                bias += lr * C * y[i]
            else:
                weights = (1 - lr) * weights
    return weights, bias

def dual_svm_objective(alpha, X, y, C):
    w = np.dot(alpha * y, X)
    hinge_loss = np.maximum(0, 1 - np.dot(X, w))
    return C * np.sum(hinge_loss) + 0.5 * np.dot(w, w)

C_values = [100 / 873, 500 / 873, 700 / 873]
primal_results = []
dual_results = []

for C in C_values:
    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    w_primal, b_primal = svm_primal(X_train, y_train, C, initial_lr=0.1, decay_rate=0.01, epochs=100)


    train_pred_primal = np.sign(np.dot(X_train, w_primal) + b_primal)
    test_pred_primal = np.sign(np.dot(test_data.iloc[:, :-1].values, w_primal) + b_primal)
    primal_results.append({
        "C": C,
        "weights": w_primal,
        "bias": b_primal,
        "train_error": np.mean(train_pred_primal != y_train),
        "test_error": np.mean(test_pred_primal != test_data.iloc[:, -1].values)
    })

    alpha_init = np.zeros(len(X_train))
    result = minimize(
        fun=dual_svm_objective,
        x0=alpha_init,
        args=(X_train, y_train, C),
        bounds=[(0, C) for _ in range(len(X_train))]
    )
    alpha_opt = result.x

    w_dual = np.dot(alpha_opt * y_train, X_train)
    b_dual = np.mean(y_train - np.dot(X_train, w_dual))  # Average over support vectors
    train_pred_dual = np.sign(np.dot(X_train, w_dual) + b_dual)
    test_pred_dual = np.sign(np.dot(test_data.iloc[:, :-1].values, w_dual) + b_dual)
    dual_results.append({
        "C": C,
        "weights": w_dual,
        "bias": b_dual,
        "train_error": np.mean(train_pred_dual != y_train),
        "test_error": np.mean(test_pred_dual != test_data.iloc[:, -1].values)
    })

for primal, dual in zip(primal_results, dual_results):
    C = primal["C"]
    weight_diff = np.linalg.norm(primal["weights"] - dual["weights"])
    bias_diff = abs(primal["bias"] - dual["bias"])
    train_error_diff = abs(primal["train_error"] - dual["train_error"])
    test_error_diff = abs(primal["test_error"] - dual["test_error"])

    print(f"C: {C}")
    print("Primal SVM:")
    print("  Weights:", primal["weights"])
    print("  Bias:", primal["bias"])
    print("  Train Error:", primal["train_error"])
    print("  Test Error:", primal["test_error"])
    print("Dual SVM:")
    print("  Weights:", dual["weights"])
    print("  Bias:", dual["bias"])
    print("  Train Error:", dual["train_error"])
    print("  Test Error:", dual["test_error"])
    print("Differences:")
    print("  Weight Difference:", weight_diff)
    print("  Bias Difference:", bias_diff)
    print("  Train Error Difference:", train_error_diff)
    print("  Test Error Difference:", test_error_diff)
    print("=" * 30)
