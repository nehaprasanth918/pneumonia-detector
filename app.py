from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("pneumonia_cnn_model_fixed.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # convert to grayscale
    image = image.resize((150, 150))  # resize to match model input
    img_array = np.array(image) / 255.0  # normalize
    img_array = img_array.reshape(1, 150, 150, 1)  # reshape for model
    return img_array

# Home route (just for testing)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(file.stream)
    processed = preprocess_image(image)

    pred = model.predict(processed)[0][0]  # single sigmoid output value

    if pred > 0.5:
        result = "Normal"      # Class 1
        confidence = pred
    else:
        result = "Pneumonia"   # Class 0
        confidence = 1 - pred

    return jsonify({
        "prediction": result,
        "confidence": float(confidence)
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
