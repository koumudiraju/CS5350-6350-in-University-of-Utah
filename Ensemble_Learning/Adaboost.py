import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_file = "DataSets/Bank/train.csv"
test_file = "DataSets/Bank/test.csv"

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
data_types = {
    'age': float,
    'job': str,
    'marital': str,
    'education': str,
    'default': str,
    'balance': float,
    'housing': str,
    'loan': str,
    'contact': str,
    'day': float,
    'month': str,
    'duration': float,
    'campaign': float,
    'pdays': float,
    'previous': float,
    'poutcome': str,
    'y': str
}
train_data = pd.read_csv(train_file, names=columns, dtype=data_types)
X_train = train_data.drop('y', axis=1).values
y_train = train_data['y'].apply(lambda label: 1 if label == 'yes' else -1).values

test_data = pd.read_csv(test_file, names=columns, dtype=data_types)
X_test = test_data.drop('y', axis=1).values
y_test = test_data['y'].apply(lambda label: 1 if label == 'yes' else -1).values

def convert_to_numeric(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        if not np.issubdtype(col.dtype, np.number):
            col = pd.to_numeric(col, errors='coerce')
            col[np.isnan(col)] = 0
        X[:, i] = col
    return X

def calculate_gain(X, y, feature_idx, threshold, sample_weights):
    n = len(y)
    left_group = X[:, feature_idx] <= threshold
    right_group = X[:, feature_idx] > threshold

    weight_left = np.sum(sample_weights[left_group])
    weight_right = np.sum(sample_weights[right_group])

    if weight_left == 0 or weight_right == 0:
        return 0

    entropy_left = -np.sum(sample_weights[left_group] * np.log2(sample_weights[left_group] / weight_left))
    entropy_right = -np.sum(sample_weights[right_group] * np.log2(sample_weights[right_group] / weight_right))

    total_entropy = (weight_left / n) * entropy_left + (weight_right / n) * entropy_right
    return total_entropy

def determine_best_split(X, y, sample_weights):
    num_features = X.shape[1]
    best_threshold = None
    best_feature = None
    lowest_entropy = float('inf')

    for feature_idx in range(num_features):
        distinct_values = np.unique(X[:, feature_idx])
        candidate_thresholds = (distinct_values[:-1] + distinct_values[1:]) / 2

        for threshold in candidate_thresholds:
            entropy = calculate_gain(X, y, feature_idx, threshold, sample_weights)

            if entropy < lowest_entropy:
                lowest_entropy = entropy
                best_threshold = threshold
                best_feature = feature_idx

    return best_feature, best_threshold

def train_stump(X, y, sample_weights):
    best_error = float('inf')
    best_stump = None
    best_labels = None

    feature, threshold = determine_best_split(X, y, sample_weights)
    
    for label_left in [-1, 1]:
        stump_predictions = np.where(X[:, feature] <= threshold, label_left, -label_left)
        error = np.sum(sample_weights * (stump_predictions != y))

        if error < best_error:
            best_error = error
            best_stump = (feature, threshold)
            best_labels = (label_left, -label_left)

    return best_stump, best_labels

def adaboost(X, y, num_iterations, learning_rate):
    X = convert_to_numeric(X)
    y = y.astype(float)
    n_samples = len(y)
    sample_weights = np.ones(n_samples) / n_samples
    alphas = []
    stumps = []
    train_errors = []
    test_errors = []
    stump_errors = []

    for i in range(num_iterations):
        (feature, threshold), (label_left, label_right) = train_stump(X, y, sample_weights)
        stump_predictions = np.where(X[:, feature] <= threshold, label_left, label_right)
        stump_error = np.sum(sample_weights * (stump_predictions != y))

        if stump_error == 0:
            alpha = 1.0
        else:
            alpha = learning_rate * np.log((1 - stump_error) / max(stump_error, 1e-10))

        alphas.append(alpha)
        stumps.append((feature, threshold, label_left, label_right))
        sample_weights *= np.exp(-alpha * y * stump_predictions)
        sample_weights /= np.sum(sample_weights)
        train_preds = adaboost_predict(X_train, alphas, stumps)
        test_preds = adaboost_predict(X_test, alphas, stumps)
        train_errors.append(1 - accuracy_score(y_train, train_preds))
        test_errors.append(1 - accuracy_score(y_test, test_preds))
        stump_errors.append(stump_error / n_samples)

    return alphas, stumps, train_errors, test_errors, stump_errors

def adaboost_predict(X, alphas, stumps):
    n_samples = X.shape[0]
    final_predictions = np.zeros(n_samples)

    for alpha, (feature, threshold, label_left, label_right) in zip(alphas, stumps):
        stump_predictions = np.where(X[:, feature] <= threshold, label_left, label_right)
        final_predictions += alpha * stump_predictions

    return np.sign(final_predictions)

num_iterations = 500
learning_rate = 0.5
alphas, stumps, train_errors, test_errors, stump_errors = adaboost(X_train, y_train, num_iterations, learning_rate)
train_error = 1 - accuracy_score(y_train, adaboost_predict(X_train, alphas, stumps))
test_error = 1 - accuracy_score(y_test, adaboost_predict(X_test, alphas, stumps))

print("Training Error:", train_error)
print("Test Error:", test_error)

# Plot the training, test, and decision stump errors over iterations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_iterations + 1), train_errors, label='Training Error', marker='o')
plt.plot(range(1, num_iterations + 1), test_errors, label='Test Error', marker='o')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Training and Test Errors')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_iterations + 1), stump_errors, label='Decision Stump Error', marker='o')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Decision Stump Errors')
plt.legend()

plt.tight_layout()
plt.show()
