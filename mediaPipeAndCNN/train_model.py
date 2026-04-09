from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.models import Sequential


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "gesture_cnn.h5"
LABELS_PATH = MODEL_DIR / "labels.npy"
EXPECTED_FEATURES = 10

X = []
y = []

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

csv_files = sorted(DATASET_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in dataset folder: {DATASET_DIR}")

for file_path in csv_files:
    gesture = file_path.stem
    df = pd.read_csv(file_path, header=None)

    if df.empty:
        continue

    if df.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{file_path.name} has {df.shape[1]} columns, expected {EXPECTED_FEATURES}."
        )

    X.extend(df.values)
    y.extend([gesture] * len(df))

if not X:
    raise ValueError("Dataset is empty. Collect gesture samples before training.")

X = np.array(X, dtype=np.float32)
y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

stratify = y if len(np.unique(y)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

model = Sequential(
    [
        Conv1D(32, kernel_size=2, activation="relu", input_shape=(EXPECTED_FEATURES, 1)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(len(np.unique(y)), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_PATH)
np.save(LABELS_PATH, le.classes_)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model saved to {MODEL_PATH}")
print(f"Labels saved to {LABELS_PATH}")
print(f"Test accuracy: {accuracy:.4f}")
