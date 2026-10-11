from flask import Flask, jsonify, request
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])#, methods=['POST']
def predictions():
    
    # N = request.form.get('name')
    user_input = []
    #83, 45, 60, 28, 70.3, 7.0, 150.9
    user_input_array = np.array(user_input)
    user_input_array = np.array([user_input_array])
    predicted_value = RF.predict(user_input_array.reshape(1, -1))
    predicted_value = {
            "Predicted": predicted_value[0]
        }
    return jsonify(predicted_value)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report
    from sklearn import metrics
    from sklearn import tree
    df = pd.read_csv('data/crop_recommendation.csv')
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    labels = df['label']
    acc = []
    model = []
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    from sklearn.ensemble import RandomForestClassifier
    global RF
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)
    try:
        app.run(port=8083, debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")
