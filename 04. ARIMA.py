import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ARIMA:
    def __init__(self, p=0, d=0, q=0):
        self.p = p
        self.d = d
        self.q = q

    def fit(self, data):
        self.data = data
        self.mu = np.mean(data)
        self.diff = np.diff(data, n=self.d)
        self.N = len(self.diff)
        self.residuals = np.zeros(self.N)
        self.params = np.zeros(self.p + self.q + 1)
        self.params[0] = self.mu

        for t in range(self.p, self.N):
            # calculate autoregressive term
            ar = np.dot(self.params[1:self.p + 1], np.flip(self.diff[t - self.p:t]))

            # calculate moving average term
            ma = np.dot(self.params[self.p + 1:], np.flip(self.residuals[t - self.q:t]))

            # calculate prediction and update residuals
            y_hat = self.params[0] + ar + ma
            self.residuals[t] = self.diff[t] - y_hat

            # update parameters using gradient descent
            grad = np.zeros(self.p + self.q + 1)
            grad[0] = -2 * np.sum(self.residuals[t - self.d:t] * self.diff[t - self.d:t]) / self.d
            grad[1:self.p + 1] = -2 * np.sum(self.residuals[t - self.p:t] * self.diff[t - self.p:t], axis=1)
            grad[self.p + 1:] = -2 * np.sum(self.residuals[t - self.q:t] * self.residuals[t - self.q:t], axis=1)
            self.params -= 0.01 * grad

    def forecast(self, n_steps):
        forecast = np.zeros(n_steps)
        for t in range(self.N, self.N + n_steps):
            ar = np.dot(self.params[1:self.p + 1], np.flip(self.diff[t - self.p:t]))
            ma = np.dot(self.params[self.p + 1:], np.flip(self.residuals[t - self.q:t]))
            forecast[t - self.N] = self.params[0] + ar + ma
            self.diff = np.append(self.diff, forecast[t - self.N] - self.mu)
        return forecast
