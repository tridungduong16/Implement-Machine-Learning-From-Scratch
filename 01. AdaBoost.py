# import numpy as np
#
#
# class AdaBoost:
#     def __init__(self, n_estimators=50):
#         self.n_estimators = n_estimators
#         self.models = []
#         self.alpha = []
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         w = np.full(n_samples, 1 / n_samples)
#
#         for i in range(self.n_estimators):
#             model = DecisionTreeClassifier(max_depth=1)
#             model.fit(X, y, sample_weight=w)
#             y_pred = model.predict(X)
#
#             error = np.sum(w[y_pred != y])
#             alpha = 0.5 * np.log((1 - error) / error)
#             w = w * np.exp(-alpha * y * y_pred)
#             w = w / np.sum(w)
#
#             self.models.append(model)
#             self.alpha.append(alpha)
#
#     def predict(self, X):
#         preds = np.zeros(len(X))
#         for i, model in enumerate(self.models):
#             preds += self.alpha[i] * model.predict(X)
#
#         return np.sign(preds)


from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score













class AdaBoost:
    def __init__(self,
                 number_of_iterations,
                 X,
                 y):
        self.number_of_iterations = number_of_iterations
        self.X = X
        self.y = y

    def train(self):
        stumps = []
        n_sample, n_features = self.X.shape
        sample_weight = [1/n_sample]*n_sample
        for i in range(self.number_of_iterations):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(self.X, self.y, sample_weight=sample_weight)
            y_pred = stump.preditc(self.X)
            total_error = np.sum(y=y_pred)
            amount_of_say = 1/2*np.log((1-total_error)/total_error)
            sample_weight = sample_weight*np.exp(amount_of_say)
            sample_weight = sample_weight / sum(sample_weight)
            stumps.append(stump)


if __name__ == "__main__":
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    feature_names = iris.feature_names
    AdaBoost(number_of_iterations=100)
