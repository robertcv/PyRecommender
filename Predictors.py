import numpy as np

class RandomPredictor(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.predictions = None

    def fit(self, X):
        self.predictions = np.random.randint(self.min, self.max, size=X.shape)

    def predict(self, user_ind):
        return self.predictions[user_ind]

class AveragePredictor(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.predictions = None

    def fit(self, X):
        self.predictions = np.random.randint(self.min, self.max, size=X.shape)

    def predict(self, user_ind):
        return self.predictions[user_ind]
