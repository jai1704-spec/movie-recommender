import numpy as np

def pearson(a, b):
    mask = (a > 0) & (b > 0)
    if np.sum(mask) == 0:
        return 0

    a, b = a[mask], b[mask]
    a_mean, b_mean = np.mean(a), np.mean(b)

    num = np.sum((a - a_mean)*(b - b_mean))
    den = np.sqrt(np.sum((a - a_mean)**2)) * np.sqrt(np.sum((b - b_mean)**2))

    return num/den if den != 0 else 0


class UserCF:
    def __init__(self, R):
        self.R = R.fillna(0).values

    def compute_similarity(self):
        n = self.R.shape[0]
        self.sim = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                self.sim[i][j] = pearson(self.R[i], self.R[j])

    def predict(self):
        return np.dot(self.sim, self.R) / (np.abs(self.sim).sum(axis=1, keepdims=True) + 1e-8)