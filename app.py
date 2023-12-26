from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
CORS(app)

# Load the Iris classifier model
iris_model = pickle.load(open('models/iris_classifier_knn_model.sav', 'rb'))

# Load the digit classification model
digit_model = load_model('models/Image-classifcation-handwritten-digits.h5')

@app.route('/')
def home():
    return "Use /iris-predict for Iris flower prediction and /digit-predict for digit classification."

@app.route('/iris-predict', methods=['POST'])
def predict_iris():
    try:
        data = request.get_json()

        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])

        prediction = iris_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        result = {"result": species[prediction]}

        return jsonify(result)
    except Exception as e:
        error_message = {"error": str(e)}
        return jsonify(error=error_message)

@app.route('/digit-predict', methods=['POST'])
def predict_digit():
    try:
        # Save the image file temporarily
        image_file = request.files['image']
        image_path = 'temp_image.png'  # Save the file temporarily
        image_file.save(image_path)

        # Load and preprocess the image
        image = img_to_array(load_img(image_path, target_size=(28, 28, 1), grayscale=True))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        # Make predictions
        prediction = digit_model.predict(image)
        predicted_class = np.argmax(prediction)

        # Remove the temporary image file
        os.remove(image_path)

        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        error_message = {"error": str(e)}
        return jsonify(error=error_message)
    
if __name__ == '__main__':
    app.run(debug=True)
