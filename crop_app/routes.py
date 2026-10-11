from flask import Blueprint, jsonify, render_template, request, session

from .ml import CROP_QUESTIONS, PRICE_QUESTIONS, predict_crop, predict_price

main_bp = Blueprint("main", __name__)

WELCOME_MESSAGE = (
    "<strong>🌾 Welcome to the Crop Recommendation Chatbot! 🌾</strong><br>"
    "I help you choose the best crop for your land and estimate market price trends.<br><br>"
    "Choose an option:<br>"
    "1. Crop Recommendation<br>"
    "2. Price Prediction<br><br>"
    "Example: type <strong>1</strong> or <strong>2</strong> to begin."
)


@main_bp.route("/")
def index():
    session.clear()
    return render_template("chat.html", welcome_message=WELCOME_MESSAGE)


@main_bp.route("/get", methods=["POST"])
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
            prediction = predict_crop(answers)
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
            payload = {
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
            prediction = predict_price(payload)
            session.clear()
            return jsonify({
                "result": f"Predicted crop price: <strong>₹{prediction:,.2f}</strong> per unit.<br>Would you like another recommendation?"
            })
        except Exception:
            session.clear()
            return jsonify({"result": "I could not process the price prediction request. Please check your values and try again."})

    return jsonify({"result": WELCOME_MESSAGE})
