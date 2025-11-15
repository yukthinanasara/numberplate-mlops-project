import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------
# Load model and encoders
# -------------------------
model = load_model("plate_vehicle_resnet_aug_model.h5")
plate_encoder = joblib.load("plate_encoder.pkl")
vehicle_encoder = joblib.load("vehicle_encoder.pkl")

# -------------------------
# Preprocessing function
# -------------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# Live webcam prediction
# -------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_input = preprocess(frame)
    plate_pred, veh_pred = model.predict(img_input)

    plate = plate_encoder.inverse_transform([np.argmax(plate_pred)])[0]
    veh = vehicle_encoder.inverse_transform([np.argmax(veh_pred)])[0]

    # Display predictions on screen
    cv2.putText(frame, f"Plate: {plate}, Vehicle: {veh}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

