# 🎬 Movie Recommendation System (Multi-Approach)

## 📌 Overview

This project implements and compares multiple recommendation techniques:

* User-Based Collaborative Filtering (Pearson Similarity)
* Item-Based Collaborative Filtering (Cosine Similarity)
* Matrix Factorization 

---

## 📊 Dataset

* MovieLens dataset from kaggle (movies.csv + ratings.csv)

---

## ⚙️ Methods & Workflows

**Steps:**
### 🔹 1. User-Based CF (Pearson)

**Steps:**

1. Build user-item matrix
2. Compute Pearson similarity between users
3. Predict ratings using weighted average

**Pros:**

* Handles user bias
* Personalized

**Cons:**

* Slow for large datasets

---

### 🔹 2. Item-Based CF (Cosine)

**Steps:**

1. Transpose user-item matrix
2. Compute cosine similarity between items
3. Predict ratings

**Pros:**

* More stable than user-based
* Scales better

---

### 🔹 3. Matrix Factorization

**Steps:**

1. Initialize latent factors
2. Use SGD to minimize error
3. Predict full matrix

**Pros:**

* Best performance
* Scalable

**Cons:**

* More complex

---

## 📈 Evaluation Metrics

* RMSE (error)

---

## 🚀 How to Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## 📊 Expected Comparison

| Method  | Accuracy | Speed  | Scalability |
| ------- | -------- | ------ | ----------- |
| User CF | Medium   | Slow   | Low         |
| Item CF | Medium   | Medium | Medium      |
| MF      | High     | Medium | High        |



---

## 🧠 Key Learnings

### 🔹 Cosine vs Pearson Similarity

Cosine similarity measures the angle between vectors and focuses on direction, making it suitable for feature-based comparisons like TF-IDF. However, it does not account for differences in user rating scales. Pearson correlation, on the other hand, subtracts the mean from ratings and captures how users deviate from their average behavior, making it more effective for collaborative filtering where user bias exists.

---

### 🔹 Handling Sparse Matrices

User-item interaction data is highly sparse, meaning most entries are missing. Iterating over all entries is inefficient, so focusing only on non-zero values significantly reduces computation. Sparse representations also help reduce memory usage and improve scalability.

---

### 🔹 Trade-offs in Recommender Systems

Different approaches have different strengths. User-based collaborative filtering is personalized but does not scale well with large datasets. Item-based filtering offers a balance between scalability and accuracy. Matrix factorization provides the best performance but is more complex and computationally intensive. Choosing the right method depends on the use case and system constraints.

---

### 🔹 Cold Start Problem

Recommender systems struggle when new users or items have no prior data. Content-based methods help mitigate this by using item features, while collaborative methods require sufficient interaction data to perform well.

---

### 🔹 Importance of Evaluation Metrics

Metrics like RMSE measure prediction accuracy, while Precision@K evaluates the quality of top recommendations. A good recommender system should balance both numerical accuracy and user satisfaction.

---

### 🔹 Scalability Considerations

Naive implementations of similarity or matrix factorization can be slow for large datasets. Optimizations like vectorization, sparse matrix operations, and efficient iteration over non-zero entries are essential for building real-world systems.
