# AgriSmart AI

A modern Flask-based crop recommendation and price prediction chatbot for farmers and agriculture enthusiasts.

## Overview
This project helps users:
- choose the best crop based on soil and weather parameters
- estimate crop market price trends using historical data
- interact through a clean, responsive chatbot UI

## Features
- Crop recommendation using a Random Forest classifier
- Market price prediction using a Random Forest regressor
- Step-by-step chatbot flow for guided input
- Modern UI built with HTML, CSS, and JavaScript
- Responsive design for desktop and mobile screens

## Tech Stack
- Python
- Flask
- scikit-learn
- Pandas
- NumPy
- HTML/CSS/JavaScript

## Project Structure
- `app.py` — compatibility entry point for the Flask app
- `run.py` — recommended startup script
- `crop_app/__init__.py` — app factory and package setup
- `crop_app/ml.py` — ML model loading and prediction logic
- `crop_app/routes.py` — route definitions and chat flow
- `templates/chat.html` — chatbot frontend
- `static/style.css` — modern styling
- `data/` — crop recommendation and price prediction datasets

## Run Locally
1. Install dependencies.
2. Start the Flask app:
   ```bash
   python run.py
   ```
3. Open the browser and visit:
   ```bash
   http://localhost:5000
   ```

## Usage Flow
1. Select `1` for crop recommendation or `2` for price prediction
2. Enter the required values when prompted
3. Receive the AI-generated output in the chat interface

## Notes
This app is intended for educational and prototype use. For real-world deployment, it should be validated against local agronomy and market conditions before being used in production decisions.
