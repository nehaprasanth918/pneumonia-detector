from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # 🔹 Added for CORS
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import traceback
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # 🔹 Enable CORS for all routes

# Load your trained CNN model
try:
    model = tf.keras.models.load_model("pneumonia_cnn_model_fixed.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    traceback.print_exc()

# Image preprocessing function
def preprocess_image(image):
    image = image.convert('L')  # Grayscale
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 150, 150, 1)
    return img_array

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Pneumonia prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            print("⚠️ No image found in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        try:
            image = Image.open(file.stream)
        except UnidentifiedImageError:
            print("⚠️ Uploaded file is not a valid image")
            return jsonify({"error": "Invalid image file"}), 400

        processed = preprocess_image(image)
        pred = model.predict(processed)[0][0]
        print(f"🔍 Raw model prediction: {pred:.4f}")

        if pred > 0.5:
            result = "Normal"
            confidence = pred
        else:
            result = "Pneumonia"
            confidence = 1 - pred

        print(f"✅ Prediction: {result}, Confidence: {confidence:.4f}")
        return jsonify({
            "prediction": result,
            "confidence": round(float(confidence), 4)
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# LLM chat route using Ollama HTTP API
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        print("⚠️ No chat message provided")
        return jsonify({"error": "No message provided"}), 400

    try:
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json={
                "model": "gemma:2b",
                "prompt": user_message,
                "stream": False
            },
            timeout=60
        )

        if response.status_code != 200:
            print("❌ Ollama API error:", response.text)
            return jsonify({"error": "Ollama API error", "details": response.text}), 500

        reply = response.json().get("response", "").strip()
        print(f"🧠 LLM reply: {reply}")
        return jsonify({"reply": reply})

    except requests.exceptions.Timeout:
        print("❌ Ollama timed out")
        return jsonify({"error": "Ollama timed out"}), 500
    except Exception as e:
        print("❌ Ollama request failed:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Ollama call failed: {str(e)}"}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
