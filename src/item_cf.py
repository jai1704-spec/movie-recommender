import numpy as np

class ItemCF:
    def __init__(self, R):
        self.R = R.fillna(0).values.T  # item-based

    def compute_similarity(self):
        norm = np.linalg.norm(self.R, axis=1, keepdims=True)
        self.sim = np.dot(self.R/norm, (self.R/norm).T)

    def predict(self):
        return np.dot(self.sim, self.R).T / (np.abs(self.sim).sum(axis=1) + 1e-8)