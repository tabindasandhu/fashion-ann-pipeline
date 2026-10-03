"""Evaluate stage: load the trained model and test set, and write metrics.json."""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend: write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    model = keras.models.load_model(os.path.join(MODELS_DIR, "model.h5"))
    test_images = np.load(os.path.join(PROCESSED_DIR, "test_images.npy"))
    test_labels = np.load(os.path.join(PROCESSED_DIR, "test_labels.npy"))

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

    predicted_labels = np.argmax(model.predict(test_images, verbose=0), axis=1)
    cm = confusion_matrix(test_labels, predicted_labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Fashion-MNIST test confusion matrix (accuracy {test_accuracy:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    # newline="\n": keep LF on Windows so the git-tracked file matches dvc.lock's hash.
    with open("metrics.json", "w", newline="\n") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_accuracy:.4f}")
    print(f"Saved metrics.json and {MODELS_DIR}/confusion_matrix.png")


if __name__ == "__main__":
    main()
