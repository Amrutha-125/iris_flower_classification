from flask import Flask, request, jsonify, send_from_directory
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
model = joblib.load("backend/iris_model.pkl")  

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result = species[prediction[0]]
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
