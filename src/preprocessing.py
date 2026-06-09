import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(path):

    df = pd.read_csv(path)

    X = df[['Age',
            'AnnualIncome',
            'PurchaseFrequency',
            'LastPurchaseDays']]

    y = df['LikelyBuyer']

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, y