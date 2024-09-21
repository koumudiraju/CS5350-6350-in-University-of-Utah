import numpy as np
import pandas as pd
from collections import Counter

def replace_unknown_with_majority(data):
    for col in range(len(data[0])):
        values = [row[col] for row in data if row[col] != 'unknown']
        majority_value = Counter(values).most_common(1)[0][0]
        
        for row in data:
            if row[col] == 'unknown':
                row[col] = majority_value
    return data

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

numerical_columns = [0, 5, 9, 11, 12, 13, 14]  
bank_train_data = binarize_numerical_features(bank_train_df.values.tolist(), numerical_columns)
bank_test_data = binarize_numerical_features(bank_test_df.values.tolist(), numerical_columns)

bank_train_data_filled = replace_unknown_with_majority(bank_train_data)
bank_test_data_filled = replace_unknown_with_majority(bank_test_data)

bank_train_errors_filled = {
    'entropy': [],
    'gini': [],
    'majority_error': []
}

bank_test_errors_filled = {
    'entropy': [],
    'gini': [],
    'majority_error': []
}

for depth in range(1, 17):
    # Criterion: Entropy
    decision_tree_entropy_filled = DecisionTreeID3(criterion='entropy', max_depth=depth)
    decision_tree_entropy_filled.fit(bank_train_data_filled)
    predictions_train_entropy_filled = decision_tree_entropy_filled.predict(bank_train_data_filled)
    predictions_test_entropy_filled = decision_tree_entropy_filled.predict(bank_test_data_filled)
    bank_train_errors_filled['entropy'].append(calculate_error(predictions_train_entropy_filled, [row[-1] for row in bank_train_data_filled]))
    bank_test_errors_filled['entropy'].append(calculate_error(predictions_test_entropy_filled, [row[-1] for row in bank_test_data_filled]))
    
    # Criterion: Gini Index
    decision_tree_gini_filled = DecisionTreeID3(criterion='gini', max_depth=depth)
    decision_tree_gini_filled.fit(bank_train_data_filled)
    predictions_train_gini_filled = decision_tree_gini_filled.predict(bank_train_data_filled)
    predictions_test_gini_filled = decision_tree_gini_filled.predict(bank_test_data_filled)
    bank_train_errors_filled['gini'].append(calculate_error(predictions_train_gini_filled, [row[-1] for row in bank_train_data_filled]))
    bank_test_errors_filled['gini'].append(calculate_error(predictions_test_gini_filled, [row[-1] for row in bank_test_data_filled]))
    
    # Criterion: Majority Error
    decision_tree_majority_filled = DecisionTreeID3(criterion='majority_error', max_depth=depth)
    decision_tree_majority_filled.fit(bank_train_data_filled)
    predictions_train_majority_filled = decision_tree_majority_filled.predict(bank_train_data_filled)
    predictions_test_majority_filled = decision_tree_majority_filled.predict(bank_test_data_filled)
    bank_train_errors_filled['majority_error'].append(calculate_error(predictions_train_majority_filled, [row[-1] for row in bank_train_data_filled]))
    bank_test_errors_filled['majority_error'].append(calculate_error(predictions_test_majority_filled, [row[-1] for row in bank_test_data_filled]))

bank_results_filled_df = pd.DataFrame({
    'Max Depth': range(1, 17),
    'Train Error (Entropy)': bank_train_errors_filled['entropy'],
    'Test Error (Entropy)': bank_test_errors_filled['entropy'],
    'Train Error (Gini)': bank_train_errors_filled['gini'],
    'Test Error (Gini)': bank_test_errors_filled['gini'],
    'Train Error (Majority Error)': bank_train_errors_filled['majority_error'],
    'Test Error (Majority Error)': bank_test_errors_filled['majority_error']
})
print(bank_results_filled_df)
