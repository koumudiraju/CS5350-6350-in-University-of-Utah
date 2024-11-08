import pandas as pd
import numpy as np
import os

train_path = 'DataSets/train.csv'
test_path = 'DataSets/test.csv'

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values


def perceptron(X, y, epochs=10, learning_rate=1.0):
    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            if y[i] * (np.dot(w, X[i])) <= 0:
                w += learning_rate * y[i] * X[i]
                b += learning_rate * y[i]

    return w, b

learning_rate = 0.1  # Adjust this value as needed
weight_vector, bias = perceptron(X_train, y_train, learning_rate=learning_rate)

def predict(X, w, b):
    return np.sign(np.dot(X, w))

y_pred = predict(X_test, weight_vector, bias)
prediction_error = np.mean(y_pred != y_test)

print("Learned Weight Vector:", weight_vector)
print("Bias:", bias)
print("Average Prediction Error on Test Dataset:", prediction_error)
