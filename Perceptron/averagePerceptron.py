import pandas as pd
import numpy as np


train_path = 'DataSets/train.csv'
test_path = 'DataSets/test.csv'

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values


def average_perceptron(X_train, y_train, max_epochs=10, learning_rate=1.0):
    # Initialize weight vector and the cumulative weight vector for averaging
    weight_vector = np.zeros(X_train.shape[1])
    average_weight = np.zeros(X_train.shape[1])
  
    for epoch in range(max_epochs):
        for i, x in enumerate(X_train):
            prediction = np.sign(np.dot(weight_vector, x))
            prediction = -1 if prediction == 0 else prediction

            if prediction * y_train[i] <= 0:  # Update on misclassification
                weight_vector += learning_rate * y_train[i] * x
            average_weight += weight_vector

    averaged_weight_vector = average_weight / (max_epochs * len(X_train))
    return averaged_weight_vector

learning_rate = 0.1
max_epochs = 10
averaged_weight_vector = average_perceptron(X_train, y_train, max_epochs, learning_rate)

def predict(X, w):
    return np.sign(np.dot(X, w))
y_pred = predict(X_test, averaged_weight_vector)
average_test_error = np.mean(y_pred != y_test)
print("Learned Averaged Weight Vector:", averaged_weight_vector)
print("Average Prediction Error on Test Data:", average_test_error)
