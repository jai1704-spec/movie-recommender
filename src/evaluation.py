import numpy as np

def rmse(true, pred):
    mask = true > 0
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))

def precision_at_k(pred, true, k=5):
    precisions = []

    for i in range(pred.shape[0]):
        top_k = np.argsort(pred[i])[-k:]
        relevant = true[i][top_k] > 0
        precisions.append(np.mean(relevant))

    return np.mean(precisions)