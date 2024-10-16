import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Data
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
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
    'label': str
}
train_path = "datasets/bank/train.csv"
test_path = "datasets/bank/test.csv"
train_data = pd.read_csv(train_path, names=column_names, dtype=data_types)
X_train_data = train_data.drop('label', axis=1).values
y_train_data = train_data['label'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

test_data = pd.read_csv(test_path, names=column_names, dtype=data_types)
X_test_data = test_data.drop('label', axis=1).values
y_test_data = test_data['label'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

# Helper function to calculate entropy
def entropy_calc(value_counts):
    total_count = sum(value_counts)
    entropy_value = 0
    for count in value_counts:
        prob = (count / total_count)
        if prob != 0:
            entropy_value -= prob * np.log2(prob)
    return entropy_value

# Helper function to calculate information gain
def info_gain_calc(X, Y, feature):
    _, class_counts = np.unique(Y, return_counts=True)
    parent_entropy = entropy_calc(class_counts)
    conditional_entropy = 0
    feature_values = np.unique(X[:, feature])
    for value in feature_values:
        value_indices = np.where(X[:, feature] == value)[0]
        _, value_class_counts = np.unique(Y[value_indices], return_counts=True)
        conditional_entropy += (len(value_indices) / len(Y)) * entropy_calc(value_class_counts)
    information_gain = parent_entropy - conditional_entropy
    return information_gain

# Function to build a decision tree (without classes)
def build_tree(X, Y, max_depth, current_depth=0):
    if current_depth >= max_depth or len(np.unique(Y)) == 1:
        vals, counts = np.unique(Y, return_counts=True)
        return vals[np.argmax(counts)]
    
    best_feature = None
    max_info_gain = -1
    for feature in range(X.shape[1]):
        info_gain = info_gain_calc(X, Y, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature

    if best_feature is None:
        vals, counts = np.unique(Y, return_counts=True)
        return vals[np.argmax(counts)]

    tree = {"feature": best_feature, "branches": {}}
    unique_values = np.unique(X[:, best_feature])

    for value in unique_values:
        indices = np.where(X[:, best_feature] == value)[0]
        tree["branches"][value] = build_tree(X[indices], Y[indices], max_depth, current_depth + 1)

    return tree

# Function to make predictions using the tree
def predict_tree(tree, instance):
    if not isinstance(tree, dict):
        return tree
    feature = tree["feature"]
    value = instance[feature]
    if value in tree["branches"]:
        return predict_tree(tree["branches"][value], instance)
    else:
        return 0

# Train and predict using the decision tree
def decision_tree_train(X, Y, max_depth):
    return build_tree(X, Y, max_depth)

def decision_tree_predict(tree, X):
    predictions = []
    for i in range(X.shape[0]):
        predictions.append(predict_tree(tree, X[i]))
    return np.array(predictions)

# Bagging function (without classes)
def bagged_trees_train(X, Y, num_trees, max_depth):
    bagged_trees = []
    n_samples = X.shape[0]

    for _ in range(num_trees):
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        Y_bootstrap = Y[bootstrap_indices]
        tree = decision_tree_train(X_bootstrap, Y_bootstrap, max_depth)
        bagged_trees.append(tree)

    return bagged_trees

# Function to predict using bagged trees
def predict_bagged_trees(bagged_trees, X):
    predictions = np.zeros(X.shape[0])

    for tree in bagged_trees:
        predictions += decision_tree_predict(tree, X)

    return np.sign(predictions)

# Vary the number of trees from 1 to 500
num_trees_list = range(1, 501)
train_errors_bagged = []
test_errors_bagged = []

for num_trees in num_trees_list:
    # Train bagged trees
    bagged_trees = bagged_trees_train(X_train_data, y_train_data, num_trees, max_depth=10)

    # Predictions
    y_train_pred = predict_bagged_trees(bagged_trees, X_train_data)
    y_test_pred = predict_bagged_trees(bagged_trees, X_test_data)

    # Calculate training and test errors
    train_error = 1 - accuracy_score(y_train_data, y_train_pred)
    test_error = 1 - accuracy_score(y_test_data, y_test_pred)

    train_errors_bagged.append(train_error)
    test_errors_bagged.append(test_error)

# Plot the training and test errors
plt.figure(figsize=(10, 6))
plt.plot(num_trees_list, train_errors_bagged, label='Train Error (Bagging)')
plt.plot(num_trees_list, test_errors_bagged, label='Test Error (Bagging)')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Trees (Bagging)')
plt.legend()
plt.show()
