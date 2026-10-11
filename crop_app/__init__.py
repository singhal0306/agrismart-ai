from flask import Flask

from .routes import main_bp
from .ml import load_crop_model, load_price_model


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["SECRET_KEY"] = "crop-recommendation-secret-key"

    app.register_blueprint(main_bp)

    load_crop_model()
    load_price_model()

    return app
