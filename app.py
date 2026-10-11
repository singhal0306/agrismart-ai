import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)
app.config["SECRET_KEY"] = "crop-recommendation-secret-key"

WELCOME_MESSAGE = (
    "<strong>🌾 Welcome to the Crop Recommendation Chatbot! 🌾</strong><br>"
    "I help you choose the best crop for your land and estimate market price trends.<br><br>"
    "Choose an option:<br>"
    "1. Crop Recommendation<br>"
    "2. Price Prediction<br><br>"
    "Example: type <strong>1</strong> or <strong>2</strong> to begin."
)

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

crop_model = None
price_preprocessor = None
price_scaler = None
price_regressor = None


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


load_crop_model()
load_price_model()


@app.route("/")
def index():
    session.clear()
    return render_template("chat.html", welcome_message=WELCOME_MESSAGE)


@app.route("/get", methods=["POST"])
def chat():
    msg = (request.form.get("msg") or "").strip()

    if not msg:
        return jsonify({"result": "Please type a valid message."})

    flow = session.get("flow")
    answers = session.get("answers", [])

    if flow is None:
        if msg not in {"1", "2"}:
            return jsonify({"result": "Please choose 1 for crop recommendation or 2 for price prediction."})

        session["flow"] = "crop" if msg == "1" else "price"
        session["answers"] = []
        next_question = CROP_QUESTIONS[0] if session["flow"] == "crop" else PRICE_QUESTIONS[0]
        return jsonify({"result": next_question})

    if flow == "crop":
        answers.append(msg)
        session["answers"] = answers

        if len(answers) < len(CROP_QUESTIONS):
            return jsonify({"result": CROP_QUESTIONS[len(answers)]})

        try:
            features = np.array(answers, dtype=float).reshape(1, -1)
            prediction = crop_model.predict(features)[0]
            session.clear()
            return jsonify({
                "result": f"According to my prediction, you should grow <strong>{prediction}</strong> this season.<br>Would you like any more help?"
            })
        except Exception:
            session.clear()
            return jsonify({"result": "I could not process the crop recommendation data. Please try again."})

    if flow == "price":
        answers.append(msg)
        session["answers"] = answers

        if len(answers) < len(PRICE_QUESTIONS):
            return jsonify({"result": PRICE_QUESTIONS[len(answers)]})

        try:
            data = pd.DataFrame([
                {
                    "State": answers[0],
                    "District": answers[1],
                    "Market": answers[2],
                    "Commodity": answers[3],
                    "Variety": answers[4],
                    "Grade": answers[5],
                    "Weekday": answers[6],
                    "Min Price": float(answers[7]),
                    "Max Price": float(answers[8]),
                }
            ])

            processed = price_preprocessor.transform(data)
            scaled = price_scaler.transform(processed)
            prediction = price_regressor.predict(scaled)[0]
            session.clear()
            return jsonify({
                "result": f"Predicted crop price: <strong>₹{prediction:,.2f}</strong> per unit.<br>Would you like another recommendation?"
            })
        except Exception:
            session.clear()
            return jsonify({"result": "I could not process the price prediction request. Please check your values and try again."})

    return jsonify({"result": WELCOME_MESSAGE})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)