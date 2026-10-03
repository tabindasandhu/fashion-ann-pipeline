"""Prepare stage: load the Fashion-MNIST dataset and save the raw arrays to data/raw/."""

import os

import numpy as np
from tensorflow import keras

RAW_DIR = os.path.join("data", "raw")


def main():
    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.fashion_mnist.load_data()
    )

    os.makedirs(RAW_DIR, exist_ok=True)
    arrays = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }
    for name, array in arrays.items():
        np.save(os.path.join(RAW_DIR, f"{name}.npy"), array)
        print(f"Saved {name}: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
# screenshot demo
# second screenshot demo line
