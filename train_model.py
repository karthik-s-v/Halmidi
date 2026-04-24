import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ================= SETTINGS =================
DATASET_DIR = 'Dataset_Final'
MODEL_SAVE_NAME = 'halmidi_model.h5'
LABELS_SAVE_NAME = 'model_labels.json'
HISTORY_SAVE_NAME = 'training_history.pkl'

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
# ============================================

def train():
    # 1. CHECK IF DATA EXISTS
    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: Folder '{DATASET_DIR}' not found!")
        return

    print("🚀 Loading Data and Setting up the Brain...")

    # 2. PREPARE DATA LOADERS
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 3. SAVE LABELS
    labels = train_data.class_indices
    labels = {v: k for k, v in labels.items()}

    with open(LABELS_SAVE_NAME, 'w') as f:
        json.dump(labels, f)

    print(f"✅ Class Labels saved to {LABELS_SAVE_NAME}")

    # 4. BUILD MODEL
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. TRAIN
    print(f"🧠 Training started for {EPOCHS} epochs...")
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data
    )

    # 6. SAVE MODEL
    model.save(MODEL_SAVE_NAME)
    print(f"✅ Model saved as: {MODEL_SAVE_NAME}")

    # 7. SAVE TRAINING HISTORY (✅ FIX)
    with open(HISTORY_SAVE_NAME, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"📊 Training history saved to {HISTORY_SAVE_NAME}")
    print(f"📈 Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print("------------------------------------------------")

if __name__ == '__main__':
    train()
