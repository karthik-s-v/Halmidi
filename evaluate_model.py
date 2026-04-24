import json
import pickle
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================= CONFIG =================
DATASET_DIR = "Dataset_Final"
MODEL_PATH = "halmidi_model.h5"
LABELS_PATH = "model_labels.json"
HISTORY_PATH = "training_history.pkl"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
# =========================================

print("🔍 Loading model, labels, and training history...")

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels_dict = json.load(f)

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

# index → class label
class_names = [labels_dict[str(i)] for i in range(len(labels_dict))]

# ================= LOAD VALIDATION DATA =================
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False   # VERY IMPORTANT for confusion matrix
)

# ================= CONFUSION MATRIX =================
print("📊 Generating Confusion Matrix...")

y_pred_probs = model.predict(val_data, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(20, 18))
sns.heatmap(
    cm_normalized,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Normalized Confusion Matrix – Halmidi Script Recognition", fontsize=20)
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("True Class", fontsize=14)
plt.tight_layout()
plt.savefig("ConfusionMatrix.png", dpi=300)
plt.show()

# ================= ACCURACY GRAPH =================
print("📈 Generating Accuracy Graph...")

plt.figure(figsize=(8, 6))
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AccuracyGraph.png", dpi=300)
plt.show()

# ================= FINAL ACCURACY =================
final_val_acc = history["val_accuracy"][-1] * 100
print(f"\n✅ Final Validation Accuracy: {final_val_acc:.2f}%")

print("\n🎉 Evaluation complete!")
print("📁 Saved files:")
print(" - AccuracyGraph.png")
print(" - ConfusionMatrix.png")
