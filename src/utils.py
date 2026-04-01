import pandas as pd
import numpy as np

def load_data(movies_path, ratings_path):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def create_user_item_matrix(ratings, movies):
    df = pd.merge(ratings, movies, on='movieId')
    return df.pivot_table(index='userId', columns='title', values='rating')

def train_test_split(matrix, test_ratio=0.2):
    train = matrix.copy()
    test = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        idx = np.where(matrix[i] > 0)[0]
        test_idx = np.random.choice(idx, size=int(len(idx)*test_ratio), replace=False)
        train[i, test_idx] = 0
        test[i, test_idx] = matrix[i, test_idx]

    return train, test