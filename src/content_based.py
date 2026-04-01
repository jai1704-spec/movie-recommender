import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBased:
    def __init__(self, movies):
        self.movies = movies.copy()

    def fit(self):
        self.movies['genres'] = self.movies['genres'].str.replace('|', ' ')
        tfidf = TfidfVectorizer()
        self.matrix = tfidf.fit_transform(self.movies['genres']).toarray()

        norm = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.similarity = np.dot(self.matrix/norm, (self.matrix/norm).T)

    def recommend(self, title, top_n=5):
        idx = self.movies[self.movies['title'] == title].index[0]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        return [self.movies.iloc[i[0]]['title'] for i in scores]