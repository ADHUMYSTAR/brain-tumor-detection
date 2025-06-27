import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# === Paths ===
train_dir = "model/data/Training"
val_dir = "model/data/Testing"
model_path = "model/saved_models/brain_tumor_model.h5"
labels_path = "model/saved_models/class_indices.json"

# === Image Parameters ===
IMG_SIZE = 224
BATCH_SIZE = 32

# === Data Generators ===
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Create output folder if it doesn't exist ===
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(labels_path), exist_ok=True)

# === Save label-to-index mapping ===
with open(labels_path, 'w') as f:
    json.dump(train_data.class_indices, f)

# === Build and Train Model ===
model = build_model(num_classes=train_data.num_classes)
model.fit(train_data, validation_data=val_data, epochs=10)

# === Save Trained Model ===
model.save(model_path)
print(f"✅ Model saved at: {model_path}")
