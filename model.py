
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -------------------------------
# MODEL CREATION
# -------------------------------

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------
# DATA VALIDATION
# -------------------------------

def _validate_data(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if x.ndim != 3 or x.shape[1:] != (28, 28):
        raise ValueError("Images must have shape (N, 28, 28).")

    if len(x) != len(y):
        raise ValueError("Image and label counts do not match.")

    if len(x) < 2:
        raise ValueError("At least two training samples are required.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Images contain invalid values.")

    if x.min() < 0 or x.max() > 1:
        raise ValueError("Images must be normalized between 0 and 1.")

    if y.min() < 0 or y.max() > 9:
        raise ValueError("Labels must be between 0 and 9.")

    return x, y


# -------------------------------
# MODEL TRAINING
# -------------------------------

def train_model(x_train, y_train, epochs=3, verbose=0):
    x_train, y_train = _validate_data(x_train, y_train)

    model = create_model()

    validation_split = 0.1 if len(x_train) >= 20 else 0.0

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=validation_split,
        verbose=verbose
    )

    return model, history


# -------------------------------
# MODEL EVALUATION
# -------------------------------

def evaluate_model(model, x_test, y_test):
    x_test, y_test = _validate_data(x_test, y_test)

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    return float(loss), float(accuracy)


# -------------------------------
# REMOVE SUSPICIOUS SAMPLES
# -------------------------------

def create_cleaned_dataset(
    x_train,
    y_train,
    suspicious_indices
):
    x_train, y_train = _validate_data(x_train, y_train)

    suspicious_indices = np.asarray(
        suspicious_indices,
        dtype=np.int64
    ).reshape(-1)

    if len(np.unique(suspicious_indices)) != len(suspicious_indices):
        raise ValueError("Duplicate suspicious indices found.")

    if np.any(suspicious_indices < 0) or np.any(
        suspicious_indices >= len(x_train)
    ):
        raise ValueError("Invalid suspicious sample index.")

    keep_mask = np.ones(len(x_train), dtype=bool)
    keep_mask[suspicious_indices] = False

    x_clean = x_train[keep_mask]
    y_clean = y_train[keep_mask]

    if len(x_clean) < 2:
        raise ValueError("Not enough samples remain after cleaning.")

    return x_clean, y_clean


# -------------------------------
# TRAIN CLEANED MODEL
# -------------------------------

def train_cleaned_model(
    x_train,
    y_train,
    suspicious_indices,
    epochs=3
):
    x_clean, y_clean = create_cleaned_dataset(
        x_train,
        y_train,
        suspicious_indices
    )

    model, history = train_model(
        x_clean,
        y_clean,
        epochs=epochs
    )

    return model, history, x_clean, y_clean


# -------------------------------
# COMPARE MODELS
# -------------------------------

def compare_models(
    original_model,
    cleaned_model,
    x_test,
    y_test
):
    original_loss, original_accuracy = evaluate_model(
        original_model,
        x_test,
        y_test
    )

    cleaned_loss, cleaned_accuracy = evaluate_model(
        cleaned_model,
        x_test,
        y_test
    )

    return {
        "original_loss": original_loss,
        "original_accuracy": original_accuracy,
        "cleaned_loss": cleaned_loss,
        "cleaned_accuracy": cleaned_accuracy,
        "accuracy_difference": (
            cleaned_accuracy - original_accuracy
        )
    }
