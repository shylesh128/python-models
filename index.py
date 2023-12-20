from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  

model = pickle.load(open('models/iris_classifier_knn_model.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return jsonify(result=result)

@app.route('/iris-predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        result = {"result": species[prediction]}
        
        return jsonify(result)
    except Exception as e:
        error_message = {"error": str(e)}
        return jsonify(error=error_message)

if __name__ == '__main__':
    app.run(debug=True)

