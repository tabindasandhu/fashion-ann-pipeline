# fashion-ann-pipeline

An MLOps assignment that trains an Artificial Neural Network (ANN) with TensorFlow/Keras
to classify images from the Fashion-MNIST dataset (10 clothing categories, 28x28 grayscale).

## Highlights

- **Model:** fully-connected ANN built with TensorFlow/Keras.
- **Reproducible pipeline:** data preparation, training and evaluation stages are defined
  as a DVC pipeline (`dvc.yaml`) with parameters tracked in `params.yaml`, so `dvc repro`
  rebuilds everything deterministically.
- **Remote storage:** datasets and model artifacts are versioned with DVC and pushed to a
  Google Drive remote; Git tracks only code and the small `.dvc` metadata files.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows (use `source venv/bin/activate` on Linux/macOS)
pip install tensorflow "dvc[gdrive]" pyyaml scikit-learn matplotlib
```

## Usage

```bash
dvc pull      # fetch data/models from the Google Drive remote
dvc repro     # run the pipeline
```
