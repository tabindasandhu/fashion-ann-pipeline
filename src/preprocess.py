"""Preprocess stage: normalize pixel values, split train/val, and save to data/processed/."""

import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def normalize(images, mean, std):
    # Standardize using training-set mean and std
    return (images.astype("float32") - mean) / std


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train_images = np.load(os.path.join(RAW_DIR, "train_images.npy"))
    train_labels = np.load(os.path.join(RAW_DIR, "train_labels.npy"))
    test_images = np.load(os.path.join(RAW_DIR, "test_images.npy"))
    test_labels = np.load(os.path.join(RAW_DIR, "test_labels.npy"))

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=train_labels,
    )

    # Statistics come from the training split only, then are applied to every
    # split, so no information leaks in from validation or test data.
    mean = float(train_images.mean(dtype="float64"))
    std = float(train_images.std(dtype="float64"))
    print(f"Training-set pixel mean={mean:.4f}, std={std:.4f}")
    train_images = normalize(train_images, mean, std)
    val_images = normalize(val_images, mean, std)
    test_images = normalize(test_images, mean, std)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    arrays = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }
    for name, array in arrays.items():
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), array)
        print(f"Saved {name}: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
