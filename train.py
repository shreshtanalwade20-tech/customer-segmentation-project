from src.preprocessing import preprocess_data
from src.clustering import train_kmeans
from src.svm_prediction import train_svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = preprocess_data("data/mall_customers.csv")

# Train KMeans
train_kmeans(X)

# Train/Test Split for SVM
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train SVM
svm = train_svm(X_train, y_train)

# Accuracy
predictions = svm.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"SVM Accuracy: {accuracy*100:.2f}%")
print("All Models Trained Successfully!")