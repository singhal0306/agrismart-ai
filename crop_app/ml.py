import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

crop_model = None
price_preprocessor = None
price_scaler = None
price_regressor = None


CROP_QUESTIONS = [
    "Enter the Nitrogen (N) value of the soil:",
    "Enter the Phosphorus (P) value of the soil:",
    "Enter the Potassium (K) value of the soil:",
    "Enter the temperature:",
    "Enter the humidity:",
    "Enter the pH value:",
    "Enter the rainfall:",
]

PRICE_QUESTIONS = [
    "Enter the state:",
    "Enter the district:",
    "Enter the market:",
    "Enter the commodity:",
    "Enter the variety:",
    "Enter the grade:",
    "Enter the weekday:",
    "Enter the minimum price:",
    "Enter the maximum price:",
]


def load_crop_model():
    global crop_model

    crop_df = pd.read_csv(os.path.join("data", "crop_recommendation.csv"))
    crop_features = crop_df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    crop_target = crop_df["label"]

    X_train, _, y_train, _ = train_test_split(
        crop_features,
        crop_target,
        test_size=0.2,
        random_state=2,
    )

    crop_model = RandomForestClassifier(n_estimators=20, random_state=0)
    crop_model.fit(X_train, y_train)


def load_price_model():
    global price_preprocessor, price_scaler, price_regressor

    price_df = pd.read_csv(os.path.join("data", "price_prediction_dataset.csv"))
    X = price_df.drop(columns=["Modal Price"])
    y = price_df["Modal Price"]

    categorical_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade", "Weekday"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough",
    )

    X_encoded = preprocessor.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)

    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train_scaled, y_train)

    price_preprocessor = preprocessor
    price_scaler = scaler
    price_regressor = regressor


def predict_crop(values):
    if crop_model is None:
        load_crop_model()
    feature_array = np.asarray(values, dtype=float).reshape(1, -1)
    return crop_model.predict(feature_array)[0]


def predict_price(values):
    if price_preprocessor is None or price_scaler is None or price_regressor is None:
        load_price_model()

    data = pd.DataFrame([values])
    processed = price_preprocessor.transform(data)
    scaled = price_scaler.transform(processed)
    return float(price_regressor.predict(scaled)[0])
