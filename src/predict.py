'''predict.py - Flask application'''
import pickle

from flask import Flask, request, jsonify


def load(model_file):
    '''Load the pre-trained model and other necessary components'''
    with open(model_file, 'rb') as f:
        dv, model = pickle.load(f)
    return dv, model


def train(data):
    '''Process the data, make predictions using the model, and return the results'''
    data = dv.transform([data])
    prediction = model.predict_proba(data)[:, 1]
    
    results_score = 'Above Average' if float(prediction) >= 0.65 else 'Below Average'
    results = {
        'probability': float(prediction),
        'prediction': results_score}
    return results


app = Flask(__name__)
model_file = '../models/dv_model.pkl'
dv, model = load(model_file=model_file)


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    results = train(data=data)
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)