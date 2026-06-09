from src.preprocessing import preprocess_data
from src.clustering import train_kmeans
from src.svm_prediction import train_svm

X, y = preprocess_data(
    "data/mall_customers.csv"
)

train_kmeans(X)

train_svm(X, y)

print("All Models Trained Successfully!")