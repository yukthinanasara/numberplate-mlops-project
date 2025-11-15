# ==============================
# LOG PRETRAINED MODEL TO MLFLOW
# ==============================

import mlflow, mlflow.keras
from tensorflow.keras.models import load_model
import joblib
import json

mlflow.set_experiment("Plate_Vehicle_Classification")

with mlflow.start_run(run_name="Log_Pretrained_Model"):

    model = load_model("plate_vehicle_resnet_aug_model.h5")
    plate_encoder = joblib.load("plate_encoder.pkl")
    vehicle_encoder = joblib.load("vehicle_encoder.pkl")

    mlflow.log_param("epochs", 15)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("base_model", "ResNet50V2")

    mlflow.keras.log_model(model, "keras-model")

    with open("training_history.json") as f:
        hist = json.load(f)

    mlflow.log_metric("final_plate_accuracy", hist["plate_output_accuracy"][-1])
    mlflow.log_metric("final_vehicle_accuracy", hist["vehicle_output_accuracy"][-1])

    mlflow.log_artifact("training_history.json")

print("🔥 Model logged to MLflow successfully!")

