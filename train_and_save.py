# ==============================
# TRAIN AND SAVE MODEL (RUN ONCE)
# ==============================

import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import json
import joblib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -----------------------------
# 1. Load dataset paths
# -----------------------------
base_dir = "/home/yukthi-de/Desktop/TEST_CW_2/numberplate/ML_02"
aug_dir = os.path.join(base_dir, "augmentation")
os.makedirs(aug_dir, exist_ok=True)

data = []

for category in ["NIBM", "Non-NIBM"]:
    image_dir = os.path.join(base_dir, category, "images")
    annot_dir = os.path.join(base_dir, category, "Annotations")
    os.makedirs(os.path.join(aug_dir, category), exist_ok=True)

    for xml_file in os.listdir(annot_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(annot_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            vehicle_type = root.find("object/name").text
            filename = root.find("filename").text
            image_path = os.path.join(image_dir, filename)

            if os.path.exists(image_path):
                data.append([image_path, category, vehicle_type])

df = pd.DataFrame(data, columns=["image_path", "plate_type", "vehicle_type"])

# -----------------------------
# 2. Encode labels
# -----------------------------
plate_encoder = LabelEncoder()
vehicle_encoder = LabelEncoder()

df["plate_label"] = plate_encoder.fit_transform(df["plate_type"])
df["vehicle_label"] = vehicle_encoder.fit_transform(df["vehicle_type"])

joblib.dump(plate_encoder, "plate_encoder.pkl")
joblib.dump(vehicle_encoder, "vehicle_encoder.pkl")

# -----------------------------
# 3. Augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

aug_paths = []
aug_p_labels = []
aug_v_labels = []

for idx, row in df.iterrows():
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    save_dir = os.path.join(aug_dir, row["plate_type"])
    aug_iter = datagen.flow(img_array, batch_size=1)

    for i in range(5):
        aug_img = next(aug_iter)[0].astype(np.uint8)
        save_path = os.path.join(save_dir, f"{idx}_aug{i}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        aug_paths.append(save_path)
        aug_p_labels.append(row["plate_label"])
        aug_v_labels.append(row["vehicle_label"])

# -----------------------------
# 4. Combine original + augmented
# -----------------------------
all_paths = list(df["image_path"]) + aug_paths
all_plate = list(df["plate_label"]) + aug_p_labels
all_vehicle = list(df["vehicle_label"]) + aug_v_labels

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img / 255.0

X = np.array([load_image(p) for p in all_paths])
y_plate = to_categorical(all_plate)
y_vehicle = to_categorical(all_vehicle)

X_train, X_test, y_p_train, y_p_test, y_v_train, y_v_test = train_test_split(
    X, y_plate, y_vehicle, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Build Model
# -----------------------------
base = tf.keras.applications.ResNet50V2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)

plate_out = Dense(2, activation="softmax", name="plate_output")(x)
vehicle_out = Dense(len(vehicle_encoder.classes_), activation="softmax", name="vehicle_output")(x)

model = Model(inputs=base.input, outputs=[plate_out, vehicle_out])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={"plate_output": "categorical_crossentropy", "vehicle_output": "categorical_crossentropy"},
    metrics={"plate_output": "accuracy", "vehicle_output": "accuracy"},
)

# -----------------------------
# 6. Train
# -----------------------------
history = model.fit(
    X_train, {"plate_output": y_p_train, "vehicle_output": y_v_train},
    validation_data=(X_test, {"plate_output": y_p_test, "vehicle_output": y_v_test}),
    epochs=15, batch_size=16
)

# -----------------------------
# 7. Save model + history
# -----------------------------
model.save("plate_vehicle_resnet_aug_model.h5")

with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("🎉 Training complete. Model saved.")

