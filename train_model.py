from datasets import load_dataset
import tensorflow as tf
import numpy as np
import os
from collections import Counter

# ============================================================
# AgroGenius AI - Plant Disease Model Training
# Corrected training script
# ============================================================

print("==============================================")
print(" AgroGenius AI - Plant Disease Model Training")
print("==============================================")

# -----------------------------
# 1. Load PlantVillage dataset
# -----------------------------
print("\nLoading PlantVillage dataset...")

dataset = load_dataset(
    "geraldmc/plantvillage-tiny",
    revision="v0.1.0",
    split="train"
)

print("Images:", len(dataset))

# ---------------------------------------------------------
# 2. Build class-name mapping from class_idx -> class_label
#    IMPORTANT: Do NOT sort class names independently.
# ---------------------------------------------------------
label_map = {}

for example in dataset:
    idx = int(example["class_idx"])
    name = str(example["class_label"])
    label_map[idx] = name

class_indices = sorted(label_map.keys())

# Make sure labels are continuous: 0,1,2,...,N-1
expected_indices = list(range(len(class_indices)))

if class_indices != expected_indices:
    raise ValueError(
        f"class_idx values are not continuous from 0 to N-1: {class_indices}"
    )

class_names = [label_map[i] for i in class_indices]
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# -----------------------------
# 3. Configuration
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 4. Prepare images and labels
# -----------------------------
print("\nPreparing images...")

images = []
labels = []

for example in dataset:
    image = example["image"].convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Keep images in 0-1 range.
    # The model below converts 0-1 -> -1 to 1 internally,
    # which is what MobileNetV2 expects.
    image = np.array(image, dtype=np.float32) / 255.0

    images.append(image)
    labels.append(int(example["class_idx"]))

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 5. Shuffle data
# -----------------------------
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]

# -----------------------------
# 6. Train / validation split
# -----------------------------
split = int(len(X) * 0.80)

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))

# -----------------------------
# 7. Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ----------------------------------------------------
# 8. MobileNetV2 transfer-learning base
# ----------------------------------------------------
print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ----------------------------------------------------
# 9. Build model
#
# Input is 0-1.
# Rescaling layer converts it to -1..1 internally,
# so app.py can continue sending normal 0-1 images.
# ----------------------------------------------------
inputs = tf.keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3),
    name="leaf_image"
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing: [0,1] -> [-1,1]
x = tf.keras.layers.Rescaling(
    scale=2.0,
    offset=-1.0,
    name="mobilenetv2_preprocessing"
)(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="disease_prediction"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 10. Class weights
# -----------------------------
class_counts = Counter(y_train.tolist())

class_weights = {}

for class_id in range(num_classes):
    count = class_counts.get(class_id, 1)
    class_weights[class_id] = len(y_train) / (num_classes * count)

print("\nClass weights calculated.")

# -----------------------------
# 11. Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# -----------------------------
# 12. Train
# -----------------------------
print("\n==============================================")
print("Starting training...")
print("==============================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 13. Evaluate
# -----------------------------
print("\n==============================================")
print("Evaluating model...")
print("==============================================")

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# -----------------------------
# 14. Save model and class names
# -----------------------------
os.makedirs("model", exist_ok=True)

model_path = "model/plant_disease_model.keras"
classes_path = "model/class_names.npy"

model.save(model_path)

# Save EXACTLY the same order used by the Dense output layer.
np.save(
    classes_path,
    np.array(class_names, dtype=str)
)

# -----------------------------
# 15. Verify saved class mapping
# -----------------------------
saved_classes = np.load(classes_path, allow_pickle=True)

print("\n==============================================")
print("TRAINING COMPLETE!")
print("==============================================")
print("Model saved to:")
print(model_path)
print("\nClass names saved to:")
print(classes_path)
print("\nSaved class mapping:")

for i, name in enumerate(saved_classes):
    print(f"{i}: {name}")

print("\nFinal validation accuracy:")
print(f"{val_accuracy * 100:.2f}%")

print("\nIMPORTANT:")
print("The model output index and class_names.npy index are now synchronized.")
print("Do NOT manually edit class_names.npy.")
print("==============================================")
