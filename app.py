# ===============================
# FLASK API FOR MODEL INFERENCE
# ===============================

from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model + encoders
model = load_model("plate_vehicle_resnet_aug_model.h5")
plate_encoder = joblib.load("plate_encoder.pkl")
vehicle_encoder = joblib.load("vehicle_encoder.pkl")

# Preprocess function
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Root route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Plate API is running"}), 200

# PREDICT ROUTE  ⭐ FIXED ⭐
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_input = preprocess(img)

    plate_pred, veh_pred = model.predict(img_input)
    plate = plate_encoder.inverse_transform([np.argmax(plate_pred)])[0]
    veh = vehicle_encoder.inverse_transform([np.argmax(veh_pred)])[0]

    return jsonify({
        "plate_type": plate,
        "vehicle_type": veh
    }), 200

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

