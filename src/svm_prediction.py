from sklearn.svm import SVC
import joblib

def train_svm(X, y):

    svm = SVC()

    svm.fit(X, y)

    joblib.dump(
        svm,
        "models/svm_model.pkl"
    )

    return svm