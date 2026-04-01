from src.utils import load_data, create_user_item_matrix, train_test_split
from src.content_based import ContentBased
from src.user_cf import UserCF
from src.item_cf import ItemCF
from src.matrix_factorization import MatrixFactorization
from src.evaluation import rmse

def recommend_for_user(pred, matrix, user_id, k=5):
    user_idx = user_id - 1

    user_ratings = matrix.values[user_idx]
    scores = pred[user_idx].copy()

    # remove already watched
    scores[user_ratings > 0] = -1

    top_k = scores.argsort()[-k:][::-1]

    return list(matrix.columns[top_k])

movies, ratings = load_data(r"/home/jai/python_programs/python_programs/ml_project/data/movies.csv", r"/home/jai/python_programs/python_programs/ml_project/data/ratings.csv")

matrix = create_user_item_matrix(ratings, movies)
train, test = train_test_split(matrix.fillna(0).values)

# User CF 
user_cf = UserCF(matrix)
user_cf.compute_similarity()
pred_user = user_cf.predict()

# Item CF 
item_cf = ItemCF(matrix)
item_cf.compute_similarity()
pred_item = item_cf.predict()

# MF
mf = MatrixFactorization(matrix)
mf.fit()
pred_mf = mf.predict()

print("UserCF RMSE:", rmse(test, pred_user))
print("ItemCF RMSE:", rmse(test, pred_item))
print("MF RMSE:", rmse(test, pred_mf))

user_id = 1

print("UserCF Recommendations:", recommend_for_user(pred_user, matrix, user_id))
print("ItemCF Recommendations:", recommend_for_user(pred_item, matrix, user_id))
print("MF Recommendations:", recommend_for_user(pred_mf, matrix, user_id))
