# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
#
# class GradientBoostingRegressor:
#     def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.learning_rate = learning_rate
#         self.models = []
#         self.intercept = None
#
#     def fit(self, X, y):
#         # Initialize F_0 as the mean of y
#         self.intercept = np.mean(y)
#         F = np.full(len(X), self.intercept)
#
#         # Train n_estimators decision trees
#         for i in range(self.n_estimators):
#             # Compute the negative gradient
#             residuals = y - F
#             # Train a decision tree on the negative gradient
#             tree = DecisionTreeRegressor(max_depth=self.max_depth)
#             tree.fit(X, residuals)
#             # Update the predictions F with the learning rate and the predictions of the decision tree
#             F += self.learning_rate * tree.predict(X)
#             # Store the trained decision tree
#             self.models.append(tree)
#
#     def predict(self, X):
#         # Compute the predictions of all the decision trees
#         trees_predictions = np.array([tree.predict(X) for tree in self.models])
#         # Compute the sum of the predictions
#         F = np.sum(trees_predictions, axis=0)
#         # Return the final predictions
#         return F + self.intercept


