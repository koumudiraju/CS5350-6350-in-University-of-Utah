import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
dtype_dict = {
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
train_file = "DataSets/Bank/train.csv"
test_file = "DataSets/Bank/test.csv"
train_df = pd.read_csv(train_file, names=column_headers, dtype=dtype_dict)
X_train = train_df.drop('y', axis=1).values
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

test_df = pd.read_csv(test_file, names=column_headers, dtype=dtype_dict)
X_test = test_df.drop('y', axis=1).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

def calculate_entropy(counts):
    total = sum(counts)
    entropy_value = 0
    for element in counts:
        p = (element / total)
        if p != 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

def calculate_information_gain(X, Y, attribute):
    _, counts = np.unique(Y, return_counts=True)
    entropy_attribute = calculate_entropy(counts)
    entropy_parent = 0
    distinct_attr_values = np.unique(X[:, attribute])
    for val in distinct_attr_values:
        indices = np.where(X[:, attribute] == val)[0]
        _, counts = np.unique(Y[indices], return_counts=True)
        entropy_val = calculate_entropy(counts)
        entropy_parent += (len(indices) / len(Y)) * entropy_val
    info_gain = entropy_attribute - entropy_parent
    return info_gain, entropy_attribute, entropy_parent

def build_tree(X, Y, max_depth, current_depth=0):
    if current_depth >= max_depth or len(np.unique(Y)) == 1:
        vals, counts = np.unique(Y, return_counts=True)
        return vals[np.argmax(counts)]  # Return most frequent label
    
    best_feature = None
    best_info_gain = -1
    for attribute in range(X.shape[1]):
        info_gain, _, _ = calculate_information_gain(X, Y, attribute)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = attribute

    if best_feature is None:
        vals, counts = np.unique(Y, return_counts=True)
        return vals[np.argmax(counts)]  # Return most frequent label

    tree = {"feature": best_feature, "children": {}}
    unique_values = np.unique(X[:, best_feature])
    
    for value in unique_values:
        indices = np.where(X[:, best_feature] == value)[0]
        tree["children"][value] = build_tree(X[indices], Y[indices], max_depth, current_depth + 1)

    return tree

def predict_tree(tree, x):
    if not isinstance(tree, dict):
        return tree
    feature = tree["feature"]
    value = x[feature]
    if value in tree["children"]:
        return predict_tree(tree["children"][value], x)
    else:
        return 0  # Return default class (0)

def train_decision_tree(X, Y, max_depth):
    return build_tree(X, Y, max_depth)

def predict_decision_tree(tree, X):
    predictions = []
    for i in range(X.shape[0]):
        predictions.append(predict_tree(tree, X[i]))
    return np.array(predictions)

def random_forest(X, Y, num_trees, max_features, max_depth):
    trees = []
    n_samples, n_features = X.shape

    for _ in range(num_trees):
        selected_features = np.random.choice(n_features, max_features, replace=False)
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices][:, selected_features]
        Y_bootstrap = Y[bootstrap_indices]
        tree = train_decision_tree(X_bootstrap, Y_bootstrap, max_depth)
        trees.append((tree, selected_features))

    return trees

def predict_random_forest(trees, X):
    predictions = np.zeros(X.shape[0])
    for tree, selected_features in trees:
        X_subset = X[:, selected_features]
        predictions += predict_decision_tree(tree, X_subset)
    return np.sign(predictions)

num_trees_range = range(1, 5)
max_features_range = [2, 4, 6]

train_errors_rf = {2: [], 4: [], 6: []}
test_errors_rf = {2: [], 4: [], 6: []}

for max_features in max_features_range:
    for num_trees in num_trees_range:
        trees = random_forest(X_train, y_train, num_trees, max_features, max_depth=10)
        y_train_pred = predict_random_forest(trees, X_train)
        y_test_pred = predict_random_forest(trees, X_test)

        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        train_errors_rf[max_features].append(train_error)
        test_errors_rf[max_features].append(test_error)

plt.figure(figsize=(12, 6))
for max_features in max_features_range:
    plt.plot(num_trees_range, train_errors_rf[max_features], label=f'Train Error (max_features={max_features})')
    plt.plot(num_trees_range, test_errors_rf[max_features], label=f'Test Error (max_features={max_features})')

plt.xlabel('Number of Random Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Random Trees (Random Forest)')
plt.legend()
plt.show()
