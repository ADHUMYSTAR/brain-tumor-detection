
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

# Load model and class labels
MODEL_PATH = "../model/saved_models/brain_tumor_model.h5"
LABELS_PATH = "../model/saved_models/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": index_to_class[predicted_class],
        "confidence": round(confidence * 100, 2)
    }
