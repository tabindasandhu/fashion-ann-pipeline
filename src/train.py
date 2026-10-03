"""Train stage: build the ANN, train it, and save models/model.h5 and history.csv."""

import csv
import os

import numpy as np
import yaml
from tensorflow import keras

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"


def build_model(dense_units, dropout_rate, learning_rate):
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(dense_units, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history, path):
    metrics = list(history.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + metrics)
        for epoch in range(len(history[metrics[0]])):
            writer.writerow([epoch + 1] + [history[m][epoch] for m in metrics])


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train_params = params["train"]

    # Seed TF/NumPy/Python RNGs so weight init and shuffling are reproducible.
    keras.utils.set_random_seed(params["preprocess"]["seed"])

    train_images = np.load(os.path.join(PROCESSED_DIR, "train_images.npy"))
    train_labels = np.load(os.path.join(PROCESSED_DIR, "train_labels.npy"))
    val_images = np.load(os.path.join(PROCESSED_DIR, "val_images.npy"))
    val_labels = np.load(os.path.join(PROCESSED_DIR, "val_labels.npy"))

    model = build_model(
        train_params["dense_units"],
        train_params["dropout_rate"],
        train_params["learning_rate"],
    )
    model.summary()

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        verbose=2,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, "model.h5"))
    save_history(history.history, os.path.join(MODELS_DIR, "history.csv"))
    print(f"Saved {MODELS_DIR}/model.h5 and {MODELS_DIR}/history.csv")


if __name__ == "__main__":
    main()
