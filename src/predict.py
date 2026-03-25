import pickle
from flask import Flask, request, jsonify

app = Flask('churn')

## ---------------------------------------Model Loading-------------------------------------

## To load the model
model_file = "model_C=1.0.bin"
with open(model_file, 'rb') as file:
    dv, model = pickle.load(file)

## -----------------------------------------Prediction---------------------------------------

## To Predict the result
@app.route("/predict", methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn_result': bool(churn)
    }

    return jsonify(result)

## Main function
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)