from sklearn.cluster import KMeans
import joblib

def train_kmeans(X):

    kmeans = KMeans(
        n_clusters=5,
        random_state=42
    )

    clusters = kmeans.fit_predict(X)

    joblib.dump(
        kmeans,
        "models/kmeans_model.pkl"
    )

    return clusters