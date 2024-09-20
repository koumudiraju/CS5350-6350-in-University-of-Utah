import numpy as np
import pandas as pd
from collections import Counter

def binarize_numerical_features(data, numerical_columns):
    for col in numerical_columns:
        median_value = np.median([row[col] for row in data if isinstance(row[col], (int, float))])
        for row in data:
            row[col] = 1 if row[col] > median_value else 0
    return data

def entropy(labels):
    total_count = len(labels)
    label_counts = Counter(labels)
    ent = -sum((count / total_count) * np.log2(count / total_count) for count in label_counts.values())
    return ent

def gini_index(labels):
    total_count = len(labels)
    label_counts = Counter(labels)
    gini = 1 - sum((count / total_count) ** 2 for count in label_counts.values())
    return gini

def majority_error(labels):
    total_count = len(labels)
    label_counts = Counter(labels)
    most_common = label_counts.most_common(1)[0][1]
    error = (total_count - most_common) / total_count
    return error

def split_data(data, attribute_index):
    split_dict = {}
    for row in data:
        key = row[attribute_index]
        if key not in split_dict:
            split_dict[key] = []
        split_dict[key].append(row)
    return split_dict

def information_gain(data, attribute_index, labels, criterion):
    total_labels = [row[-1] for row in data]
    initial_impurity = criterion(total_labels)
    
    splits = split_data(data, attribute_index)
    weighted_impurity = sum((len(subset) / len(data)) * criterion([row[-1] for row in subset]) 
                            for subset in splits.values())
    
    return initial_impurity - weighted_impurity

class DecisionTreeID3:
    def _init_(self, criterion='entropy', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
    
    def _choose_best_attribute(self, data):
        total_labels = [row[-1] for row in data]
        
        if self.criterion == 'entropy':
            criterion_function = entropy
        elif self.criterion == 'gini':
            criterion_function = gini_index
        else:
            criterion_function = majority_error
        
        gains = [information_gain(data, i, total_labels, criterion_function) for i in range(len(data[0]) - 1)]
        best_attribute = np.argmax(gains)
        return best_attribute
    
    def _build_tree(self, data, depth=0):
        labels = [row[-1] for row in data]
        
        if len(set(labels)) == 1:
            return labels[0]
        
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(labels).most_common(1)[0][0]
        
        best_attribute = self._choose_best_attribute(data)
        
        tree = {best_attribute: {}}
        splits = split_data(data, best_attribute)
        for value, subset in splits.items():
            tree[best_attribute][value] = self._build_tree(subset, depth + 1)
        
        return tree
    
    def fit(self, data):
        self.tree = self._build_tree(data)
    
    def _predict_one(self, example, tree):
        if isinstance(tree, dict):
            attribute = next(iter(tree))
            value = example[attribute]
            if value in tree[attribute]:
                return self._predict_one(example, tree[attribute][value])
            else:
                return None
        else:
            return tree
    
    def predict(self, data):
        return [self._predict_one(example, self.tree) for example in data]

def calculate_error(predictions, true_labels):
    return 1 - (sum(1 for true, pred in zip(true_labels, predictions) if true == pred) / len(true_labels))

bank_train_file_path = 'DataSets/Bank/train.csv'
bank_test_file_path = 'DataSets/Bank/test.csv'
bank_train_df = pd.read_csv(bank_train_file_path, header=None)
bank_test_df = pd.read_csv(bank_test_file_path, header=None)

numerical_columns = [0, 5, 11, 12]  # Assuming these are the numerical columns
bank_train_data = binarize_numerical_features(bank_train_df.values.tolist(), numerical_columns)
bank_test_data = binarize_numerical_features(bank_test_df.values.tolist(), numerical_columns)

bank_train_errors = {
    'entropy': [],
    'gini': [],
    'majority_error': []
}

bank_test_errors = {
    'entropy': [],
    'gini': [],
    'majority_error': []
}

for depth in range(1, 17):
    # Criterion: Entropy
    decision_tree_entropy = DecisionTreeID3(criterion='entropy', max_depth=depth)
    decision_tree_entropy.fit(bank_train_data)
    predictions_train_entropy = decision_tree_entropy.predict(bank_train_data)
    predictions_test_entropy = decision_tree_entropy.predict(bank_test_data)
    bank_train_errors['entropy'].append(calculate_error(predictions_train_entropy, [row[-1] for row in bank_train_data]))
    bank_test_errors['entropy'].append(calculate_error(predictions_test_entropy, [row[-1] for row in bank_test_data]))
    
    decision_tree_gini = DecisionTreeID3(criterion='gini', max_depth=depth)
    decision_tree_gini.fit(bank_train_data)
    predictions_train_gini = decision_tree_gini.predict(bank_train_data)
    predictions_test_gini = decision_tree_gini.predict(bank_test_data)
    bank_train_errors['gini'].append(calculate_error(predictions_train_gini, [row[-1] for row in bank_train_data]))
    bank_test_errors['gini'].append(calculate_error(predictions_test_gini, [row[-1] for row in bank_test_data]))
    
    decision_tree_majority = DecisionTreeID3(criterion='majority_error', max_depth=depth)
    decision_tree_majority.fit(bank_train_data)
    predictions_train_majority = decision_tree_majority.predict(bank_train_data)
    predictions_test_majority = decision_tree_majority.predict(bank_test_data)
    bank_train_errors['majority_error'].append(calculate_error(predictions_train_majority, [row[-1] for row in bank_train_data]))
    bank_test_errors['majority_error'].append(calculate_error(predictions_test_majority, [row[-1] for row in bank_test_data]))

bank_results_df = pd.DataFrame({
    'Max Depth': range(1, 17),
    'Train Error (Entropy)': bank_train_errors['entropy'],
    'Test Error (Entropy)': bank_test_errors['entropy'],
    'Train Error (Gini)': bank_train_errors['gini'],
    'Test Error (Gini)': bank_test_errors['gini'],
    'Train Error (Majority Error)': bank_train_errors['majority_error'],
    'Test Error (Majority Error)': bank_test_errors['majority_error']
})

print(bank_results_df)
