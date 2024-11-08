import pandas as pd
import numpy as np

train_path = 'DataSets/train.csv'
test_path = 'DataSets/test.csv'

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

def voted_perceptron(X_train, y_train, max_epochs, learning_rate):
    # Initialize variables
    sample_count, feature_count = X_train.shape
    weight_vector = np.zeros(feature_count)  # Initialize weights to zero
    weight_history = []  # List to store unique weight vectors
    accuracy_counts = []  # List to store counts of consecutive correct predictions
    correction_count = 0  # Track index for corrections

    for epoch in range(max_epochs):
        mistakes = 0
        correct_predictions = 1  # Initialize count of correct predictions for each weight

        for idx, x in enumerate(X_train):
            predicted_label = np.sign(np.dot(weight_vector, x))
            predicted_label = -1 if predicted_label == 0 else predicted_label
            if predicted_label * y_train[idx] <= 0:
                # Adjust weights based on the learning rate
                weight_vector += learning_rate * y_train[idx] * x
                correction_count += 1
                correct_predictions = 1
                mistakes += 1
            else:
                correct_predictions += 1
        weight_history.append(weight_vector.copy())
        accuracy_counts.append(sample_count - mistakes)

    return weight_history, accuracy_counts

max_epochs = 10
learning_rate = 0.1
weight_history, accuracy_counts = voted_perceptron(X_train, y_train, max_epochs, learning_rate)

test_errors = []
for weight, count in zip(weight_history, accuracy_counts):
    errors = 0
    for idx, x in enumerate(X_test):
        predicted_label = np.sign(np.dot(weight, x))
        predicted_label = -1 if predicted_label == 0 else predicted_label
        if predicted_label != y_test[idx]:
            errors += 1
    test_errors.append(errors / len(X_test))

average_test_error = np.mean(test_errors)

for i, (weight, count) in enumerate(zip(weight_history, accuracy_counts), start=1):
    print(f"Weight Vector {i}: {weight}, Correct Count: {count}")

print(f"Average Test Error: {average_test_error:.2f}")
