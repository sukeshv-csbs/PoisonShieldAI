
import gzip
import io
import struct
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_SHAPE = (28, 28)
NUM_CLASSES = 10


# IMAGE PREPROCESSING
def _prepare_images(images):
    images = np.asarray(images)

    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    elif images.ndim == 4:
        if images.shape[-1] == 1:
            images = images[..., 0]
        elif images.shape[1] == 1:
            images = images[:, 0, :, :]
        else:
            raise ValueError("Only grayscale images are supported.")

    if images.ndim != 3 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError(
            f"Invalid image shape: {images.shape}. Expected (N, 28, 28)."
        )

    images = images.astype(np.float32)

    if not np.all(np.isfinite(images)):
        raise ValueError("Images contain invalid numerical values.")

    if images.min() < 0 or images.max() > 255:
        raise ValueError("Pixel values must be between 0 and 255.")

    if images.max() > 1:
        images /= 255.0

    return images


# LABEL PREPROCESSING
def _prepare_labels(labels):
    labels = np.asarray(labels).reshape(-1)

    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Labels must be numerical.")

    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels contain invalid values.")

    if not np.all(labels == np.floor(labels)):
        raise ValueError("Labels must be whole numbers.")

    labels = labels.astype(np.int64)

    if len(labels) == 0:
        raise ValueError("No labels found.")

    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError("Labels must be between 0 and 9.")

    return labels


# TRAIN / TEST SPLIT
def _split_dataset(images, labels, test_size=0.2):
    if len(images) != len(labels):
        raise ValueError("Image and label counts do not match.")

    if len(images) < 2:
        raise ValueError("At least two samples are required.")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(images))

    split = int(len(images) * (1 - test_size))
    split = max(1, min(split, len(images) - 1))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return (
        images[train_idx],
        labels[train_idx],
        images[test_idx],
        labels[test_idx]
    )


# NPZ LOADER
def _load_npz(uploaded_file):
    try:
        data = np.load(
            io.BytesIO(uploaded_file.getvalue()),
            allow_pickle=False
        )

        keys = list(data.files)

        def find_key(names):
            for name in names:
                if name in keys:
                    return data[name]
            return None

        x_train = find_key(["x_train", "train_images", "images"])
        y_train = find_key(["y_train", "train_labels", "labels"])
        x_test = find_key(["x_test", "test_images"])
        y_test = find_key(["y_test", "test_labels"])

        if x_train is None or y_train is None:
            raise ValueError("NPZ must contain training images and labels.")

        x_train = _prepare_images(x_train)
        y_train = _prepare_labels(y_train)

        if len(x_train) != len(y_train):
            raise ValueError("Training image and label counts differ.")

        if x_test is not None and y_test is not None:
            x_test = _prepare_images(x_test)
            y_test = _prepare_labels(y_test)

            if len(x_test) != len(y_test):
                raise ValueError("Test image and label counts differ.")

            return x_train, y_train, x_test, y_test

        if x_test is not None or y_test is not None:
            raise ValueError("Both x_test and y_test must be provided.")

        return _split_dataset(x_train, y_train)

    except Exception as e:
        raise ValueError(f"Could not load NPZ: {e}")


# CSV LOADER
def _load_csv(uploaded_file):
    try:
        df = pd.read_csv(
            io.BytesIO(uploaded_file.getvalue()),
            header=None
        )

        df = df.apply(pd.to_numeric, errors="coerce")

        if df.iloc[0].isna().any():
            df = df.iloc[1:].reset_index(drop=True)

        if df.isna().any().any():
            raise ValueError("CSV contains missing or non-numeric values.")

        if df.shape[1] != 785:
            raise ValueError("CSV must contain 784 pixels and 1 label.")

        values = df.to_numpy()

        first = values[:, 0]
        last = values[:, -1]

        def valid_label_column(column):
            return (
                len(column) > 0
                and np.all(np.isfinite(column))
                and np.all(column == np.floor(column))
                and np.all((column >= 0) & (column <= 9))
            )

        if valid_label_column(first):
            labels = first
            images = values[:, 1:]
        elif valid_label_column(last):
            labels = last
            images = values[:, :-1]
        else:
            raise ValueError("Could not identify the label column.")

        images = _prepare_images(images)
        labels = _prepare_labels(labels)

        return _split_dataset(images, labels)

    except Exception as e:
        raise ValueError(f"Could not load CSV: {e}")


# IDX / UBYTE LOADER
def _get_file_bytes(uploaded_file):
    content = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".gz"):
        content = gzip.decompress(content)

    return content


def _read_idx_pair(image_file, label_file):
    image_bytes = _get_file_bytes(image_file)
    label_bytes = _get_file_bytes(label_file)

    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError("IDX file is incomplete.")

    image_magic, image_count, rows, cols = struct.unpack(
        ">IIII", image_bytes[:16]
    )

    label_magic, label_count = struct.unpack(
        ">II", label_bytes[:8]
    )

    if image_magic != 2051 or label_magic != 2049:
        raise ValueError("Invalid IDX image or label file.")

    if rows != 28 or cols != 28:
        raise ValueError("Only 28x28 images are supported.")

    if image_count != label_count:
        raise ValueError("Image and label counts differ.")

    if len(image_bytes) != 16 + image_count * rows * cols:
        raise ValueError("Invalid IDX image file size.")

    if len(label_bytes) != 8 + label_count:
        raise ValueError("Invalid IDX label file size.")

    images = np.frombuffer(
        image_bytes, dtype=np.uint8, offset=16
    ).reshape(image_count, 28, 28)

    labels = np.frombuffer(
        label_bytes, dtype=np.uint8, offset=8
    )

    return _prepare_images(images), _prepare_labels(labels)


def _load_idx(files):
    image_files = []
    label_files = []

    for file in files:
        name = file.name.lower()

        if "images-idx3" in name:
            image_files.append(file)
        elif "labels-idx1" in name:
            label_files.append(file)

    if not image_files or not label_files:
        raise ValueError("Upload matching IDX image and label files.")

    def find_pair(keyword):
        image = next(
            (f for f in image_files if keyword in f.name.lower()),
            None
        )
        label = next(
            (f for f in label_files if keyword in f.name.lower()),
            None
        )

        if image is not None and label is not None:
            return image, label

        return None

    train_pair = find_pair("train")
    test_pair = find_pair("t10k")

    if train_pair and test_pair:
        x_train, y_train = _read_idx_pair(*train_pair)
        x_test, y_test = _read_idx_pair(*test_pair)

        return x_train, y_train, x_test, y_test

    if len(image_files) == 1 and len(label_files) == 1:
        images, labels = _read_idx_pair(
            image_files[0], label_files[0]
        )
        return _split_dataset(images, labels)

    raise ValueError(
        "Upload one matching image/label pair or all four standard "
        "MNIST train/test IDX files."
    )


# MAIN UPLOAD FUNCTION
def load_uploaded_mnist(uploaded_files):
    if not uploaded_files:
        raise ValueError("Please upload a dataset.")

    files = list(uploaded_files)

    extensions = [
        Path(file.name).suffix.lower()
        for file in files
    ]

    if len(files) == 1 and extensions[0] == ".npz":
        return _load_npz(files[0])

    if len(files) == 1 and extensions[0] == ".csv":
        return _load_csv(files[0])

    if all(ext in [".idx", ".ubyte", ".gz"] for ext in extensions):
        return _load_idx(files)

    raise ValueError(
        "Use one NPZ file, one CSV file, or matching IDX/UBYTE files."
    )


# OPTIONAL BUILT-IN MNIST LOADER
def load_mnist():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (
        _prepare_images(x_train),
        _prepare_labels(y_train),
        _prepare_images(x_test),
        _prepare_labels(y_test)
    )


# LABEL FLIPPING
def flip_labels(
    labels,
    flip_rate=0.1,
    random_state=42,
    source_class=None,
    target_class=None
):
    labels = _prepare_labels(labels).copy()

    if not 0 <= flip_rate <= 1:
        raise ValueError("Flip rate must be between 0 and 1.")

    if source_class is not None and not 0 <= source_class <= 9:
        raise ValueError("Source class must be between 0 and 9.")

    if target_class is not None and not 0 <= target_class <= 9:
        raise ValueError("Target class must be between 0 and 9.")

    rng = np.random.default_rng(random_state)

    if source_class is not None:
        candidates = np.where(labels == source_class)[0]
    else:
        candidates = np.arange(len(labels))

    count = int(len(candidates) * flip_rate)

    if flip_rate > 0 and len(candidates) > 0:
        count = max(1, count)

    count = min(count, len(candidates))

    if count == 0:
        return labels, np.array([], dtype=np.int64)

    selected = rng.choice(
        candidates,
        size=count,
        replace=False
    )

    for index in selected:
        old_label = labels[index]

        if target_class is not None:
            if target_class == old_label:
                raise ValueError(
                    "Target class must differ from the original label."
                )
            labels[index] = target_class
        else:
            choices = [
                digit for digit in range(10)
                if digit != old_label
            ]
            labels[index] = rng.choice(choices)

    return labels, selected


# POISONING SUMMARY
def summarize_poisoning(
    original_labels,
    poisoned_labels,
    poisoned_indices
):
    original_labels = _prepare_labels(original_labels)
    poisoned_labels = _prepare_labels(poisoned_labels)

    if len(original_labels) != len(poisoned_labels):
        raise ValueError("Label counts differ.")

    report = []

    for digit in range(NUM_CLASSES):
        original_count = int(np.sum(original_labels == digit))
        poisoned_count = int(np.sum(poisoned_labels == digit))

        report.append({
            "Class": digit,
            "Original Count": original_count,
            "After Poisoning": poisoned_count,
            "Difference": poisoned_count - original_count
        })

    return pd.DataFrame(report)


# CREATE CONTROLLED POISONED DATASET
def create_poisoned_dataset(
    x_train,
    y_train,
    x_test=None,
    y_test=None,
    flip_rate=0.1,
    random_state=42,
    source_class=None,
    target_class=None
):
    x_train = _prepare_images(x_train)
    y_train = _prepare_labels(y_train)

    if len(x_train) != len(y_train):
        raise ValueError("Training image and label counts differ.")

    if (x_test is None) != (y_test is None):
        raise ValueError("Provide both test images and test labels.")

    if x_test is not None:
        x_test = _prepare_images(x_test)
        y_test = _prepare_labels(y_test)

        if len(x_test) != len(y_test):
            raise ValueError("Test image and label counts differ.")

    # This is the important corrected function call.
    poisoned_labels, poisoned_indices = flip_labels(
        y_train,
        flip_rate=flip_rate,
        random_state=random_state,
        source_class=source_class,
        target_class=target_class
    )

    report = summarize_poisoning(
        y_train,
        poisoned_labels,
        poisoned_indices
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_poisoned": poisoned_labels,
        "x_test": x_test,
        "y_test": y_test,
        "poisoned_indices": poisoned_indices,
        "poisoning_report": report,
        "flip_rate": flip_rate
    }


# SAVE DATASET
def save_dataset(
    x_train,
    y_train,
    x_test,
    y_test,
    output_path="cleaned_dataset.npz"
):
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
