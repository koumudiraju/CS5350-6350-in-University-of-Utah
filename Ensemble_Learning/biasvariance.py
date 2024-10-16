import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

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
train_data_path = "DataSets/Bank/train.csv"
test_data_path = "DataSets/Bank/test.csv"
train_df = pd.read_csv(train_data_path, names=columns, dtype=data_types)
X_train = train_df.drop('y', axis=1).values
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

test_df = pd.read_csv(test_data_path, names=columns, dtype=data_types)
X_test = test_df.drop('y', axis=1).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

def compute_bias_variance(preds, actuals):
    bias_val = np.mean(preds) - np.mean(actuals)
    variance_val = np.var(preds)

    return bias_val, variance_val

def simple_decision_tree(X, Y, max_tree_depth):
    return construct_tree(X, Y, max_tree_depth)

def bagging_trees(X_train_data, y_train_data, num_trees, tree_max_depth):
    trees_collection = []
    num_samples = X_train_data.shape[0]

    for _ in range(num_trees):
        boot_indices = np.random.choice(num_samples, num_samples, replace=True)
        X_bootstrap, y_bootstrap = X_train_data[boot_indices], y_train_data[boot_indices]

        # Train a simple decision tree on the bootstrap sample
        tree_model = simple_decision_tree(X_bootstrap, y_bootstrap, tree_max_depth)
        trees_collection.append(tree_model)

    return trees_collection
  
def predict_bagged(trees, X_test_data):
    avg_preds = np.zeros(X_test_data.shape[0])

    for tree_model in trees:
        avg_preds += predict_simple_tree(tree_model, X_test_data)

    return avg_preds / len(trees)

def predict_simple_tree(tree_model, X_data):
    predictions = []
    for i in range(X_data.shape[0]):
        predictions.append(traverse_tree(tree_model, X_data[i]))
    return np.array(predictions)

def construct_tree(X, Y, max_tree_depth, current_depth=0):
    if current_depth >= max_tree_depth or len(np.unique(Y)) == 1:
        label_counts = np.unique(Y, return_counts=True)
        return label_counts[0][np.argmax(label_counts[1])]  # Return most frequent class
    
    best_feature = None
    max_info_gain = -1
    for feature_idx in range(X.shape[1]):
        info_gain, _, _ = info_gain_calc(X, Y, feature_idx)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature_idx

    if best_feature is None:
        label_counts = np.unique(Y, return_counts=True)
        return label_counts[0][np.argmax(label_counts[1])]  # Return most frequent class

    tree_structure = {"feature": best_feature, "branches": {}}
    distinct_values = np.unique(X[:, best_feature])
    
    for value in distinct_values:
        split_indices = np.where(X[:, best_feature] == value)[0]
        tree_structure["branches"][value] = construct_tree(X[split_indices], Y[split_indices], max_tree_depth, current_depth + 1)

    return tree_structure

def traverse_tree(tree_structure, x_instance):
    if not isinstance(tree_structure, dict):
        return tree_structure
    feature_idx = tree_structure["feature"]
    feature_value = x_instance[feature_idx]
    if feature_value in tree_structure["branches"]:
        return traverse_tree(tree_structure["branches"][feature_value], x_instance)
    else:
        return 0  # Default prediction

def info_gain_calc(X, Y, feature_idx):
    _, counts = np.unique(Y, return_counts=True)
    parent_entropy = entropy_calc(counts)
    conditional_entropy = 0
    unique_values = np.unique(X[:, feature_idx])
    for val in unique_values:
        indices = np.where(X[:, feature_idx] == val)[0]
        _, val_counts = np.unique(Y[indices], return_counts=True)
        conditional_entropy += (len(indices) / len(Y)) * entropy_calc(val_counts)
    info_gain_val = parent_entropy - conditional_entropy
    return info_gain_val, parent_entropy, conditional_entropy

def entropy_calc(counts):
    total_count = sum(counts)
    entropy_val = 0
    for count in counts:
        probability = (count / total_count)
        if probability != 0:
            entropy_val -= probability * np.log2(probability)
    return entropy_val

def perform_experiment(X_train_data, y_train_data, X_test_data, y_test_data, num_iterations=100, num_bagged_trees=500, tree_max_depth=10):
    single_tree_bias_vals = []
    single_tree_variance_vals = []
    bagged_tree_bias_vals = []
    bagged_tree_variance_vals = []

    for _ in range(num_iterations):
        sample_size = X_train_data.shape[0]
        selected_indices = np.random.choice(sample_size, size=1000, replace=False)
        X_sample, y_sample = X_train_data[selected_indices], y_train_data[selected_indices]

        bagged_trees = bagging_trees(X_sample, y_sample, num_bagged_trees, tree_max_depth)

        single_tree_preds = np.array([predict_simple_tree(tree, X_test_data) for tree in bagged_trees])
        avg_single_tree_preds = np.mean(single_tree_preds, axis=0)

        bias_single, var_single = compute_bias_variance(avg_single_tree_preds, y_test_data)
        single_tree_bias_vals.append(bias_single)
        single_tree_variance_vals.append(var_single)
        bagged_tree_preds = predict_bagged(bagged_trees, X_test_data)

        bias_bagged, var_bagged = compute_bias_variance(bagged_tree_preds, y_test_data)
        bagged_tree_bias_vals.append(bias_bagged)
        bagged_tree_variance_vals.append(var_bagged)

    avg_single_tree_bias = np.mean(single_tree_bias_vals)
    avg_single_tree_variance = np.mean(single_tree_variance_vals)
    avg_bagged_tree_bias = np.mean(bagged_tree_bias_vals)
    avg_bagged_tree_variance = np.mean(bagged_tree_variance_vals)

    return avg_single_tree_bias, avg_single_tree_variance, avg_bagged_tree_bias, avg_bagged_tree_variance

avg_single_tree_bias, avg_single_tree_variance, avg_bagged_tree_bias, avg_bagged_tree_variance = perform_experiment(
    X_train, y_train, X_test, y_test)

print("Average bias for single decision tree:", avg_single_tree_bias)
print("Average variance for single decision tree:", avg_single_tree_variance)
print("Average bias for bagged trees:", avg_bagged_tree_bias)
print("Average variance for bagged trees:", avg_bagged_tree_variance)
