from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class DecisionStump:
    def __init__(self):
        self.stump = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        self.stump.fit(X, y)

    def predict(self, X):
        return self.stump.predict(X)


class AdaBoost:
    def __init__(self):
        pass


stump = DecisionStump()
