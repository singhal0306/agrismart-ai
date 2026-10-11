from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

INITIAL_MESSAGE = "Hey, this is crop recommendation chatbot. I can help you pick best crop so that you can yield more efficiently and can earn more."

# App state
state = {
    "k": 1,
    "i": 0,
    "flag": 0,
    "user_input": [],
    "new_user_input": [],
}

# Model variables
RF = None
preprocessor = None
sc = None
regressor = None


def reset_state():
    state["k"] = 1
    state["i"] = 0
    state["flag"] = 0
    state["user_input"] = []
    state["new_user_input"] = []


def initialize_models():
    global RF, preprocessor, sc, regressor

    crop_df = pd.read_csv("data/crop_recommendation.csv")
    crop_features = crop_df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    crop_target = crop_df["label"]

    Xtrain, _, Ytrain, _ = train_test_split(
        crop_features,
        crop_target,
        test_size=0.2,
        random_state=2,
    )

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain, Ytrain)

    price_df = pd.read_csv("data/price_prediction_dataset.csv")
    X = price_df.drop(columns=["Modal Price"])
    y = price_df["Modal Price"]

    categorical_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade", "Weekday"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(), categorical_cols)],
        remainder="passthrough",
    )

    X_encoded = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=0,
    )

    sc = StandardScaler(with_mean=False)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)


def handle_crop_recommendation(msg):
    question_list = [
        "Enter your loaction:",
        "Enter the Nitrogen(N) content of soil: ",
        "Enter the Phosphorus(P) content of soil",
        "Enter the Potassium(K) content of soil",
        "Enter the temperature:",
        "Enter the humidity:",
        "Enter the ph value:",
        "Enter the rainfall:",
    ]

    if state["i"] < len(question_list):
        state["user_input"].append(msg)
        next_question = question_list[state["i"]]
        state["i"] += 1
        return {"result": next_question}

    state["user_input"].append(msg)
    user_input_array = np.array(state["user_input"][2:])
    user_input_array = np.array([user_input_array])
    predicted_value = RF.predict(user_input_array.reshape(1, -1))

    response = {
        "result": "According to my prediction you must grow <b>{"
        + predicted_value[0]
        + "}</b> this year.<br>Is there anything else I can help you with."
    }
    reset_state()
    return response


def handle_price_prediction(msg):
    question_list = [
        "Enter the state: ",
        "Enter the district: ",
        "Enter the market: ",
        "Enter the commodity: ",
        "Enter the variety: ",
        "Enter the grade: ",
        "Enter the Weekday: ",
        "Enter the min price: ",
        "Enter the max price: ",
    ]

    if state["i"] < len(question_list):
        state["new_user_input"].append(msg)
        next_question = question_list[state["i"]]
        state["i"] += 1
        return {"result": next_question}

    state["new_user_input"].append(msg)
    user_input = pd.DataFrame(
        {
            "State": [state["new_user_input"][1]],
            "District": [state["new_user_input"][2]],
            "Market": [state["new_user_input"][3]],
            "Commodity": [state["new_user_input"][4]],
            "Variety": [state["new_user_input"][5]],
            "Grade": [state["new_user_input"][6]],
            "Weekday": [state["new_user_input"][7]],
            "Min Price": [state["new_user_input"][8]],
            "Max Price": [state["new_user_input"][9]],
        }
    )

    user_input_preprocessed = preprocessor.transform(user_input)
    user_input_preprocessed = sc.transform(user_input_preprocessed)
    predicted_price = regressor.predict(user_input_preprocessed)

    response = {"result": f"Predicted Crop Price: {predicted_price[0]}"}
    state["new_user_input"].clear()
    reset_state()
    return response


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()

    if state["k"] == 1 and msg in {"1", "2"}:
        state["flag"] = int(msg)
        state["k"] = 0

    if state["flag"] == 1:
        return handle_crop_recommendation(msg)

    if state["flag"] == 2:
        return handle_price_prediction(msg)

    welcome_msg = (
        '🌾 Welcome to the Crop Recommendation Chatbot! 🌾<br>'
        "I am here to help you make informed decisions about the best crops to plant based on atmospheric conditions and price predictions. "
        "Whether you are a seasoned farmer or just getting started, I have got you covered. <br><br>"
        'You can ask me questions like: <br>- Press 1: "What should I plant this season?" <br>- Press 2: "Can you recommend crops with good price predictions?" <br><br>'
        "Feel free to explore and ask anything related to crop recommendations and atmospheric conditions. Let us work together to maximize your agricultural yield! <br>"
    )
    state["flag"] = 0
    return {"result": welcome_msg}


if __name__ == "__main__":
    initialize_models()
    app.run(debug=True)