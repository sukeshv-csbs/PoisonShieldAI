
import numpy as np


def calculate_risk_scores(losses):
    """
    Convert sample losses into relative percentile risk scores.
    Higher scores mean more unusual samples.
    These are NOT probabilities of poisoning.
    """

    losses = np.asarray(losses, dtype=np.float64).reshape(-1)

    if len(losses) == 0:
        return np.array([], dtype=np.float32)

    if not np.all(np.isfinite(losses)):
        raise ValueError("Loss values contain invalid numbers.")

    sorted_losses = np.sort(losses)
    left = np.searchsorted(sorted_losses, losses, side="left")
    right = np.searchsorted(sorted_losses, losses, side="right")

    scores = ((left + right) / 2) / len(losses) * 100

    return scores.astype(np.float32)


def detect_suspicious_samples(
    model,
    x_train,
    y_train,
    threshold=95,
    batch_size=256,
    detection_rate=None
):
    """
    Detect suspicious samples using prediction loss.

    threshold:
        Minimum percentile risk score.

    detection_rate:
        Optional percentage of samples to flag.
        Example: 5 means flag the top 5%.
    """

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)

    if x_train.ndim != 3 or x_train.shape[1:] != (28, 28):
        raise ValueError("Images must have shape (N, 28, 28).")

    if len(x_train) != len(y_train):
        raise ValueError("Image and label counts do not match.")

    if len(x_train) == 0:
        raise ValueError("No training samples provided.")

    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100.")

    if detection_rate is not None and not 0 < detection_rate <= 100:
        raise ValueError("Detection rate must be greater than 0 and at most 100.")

    if y_train.min() < 0 or y_train.max() > 9:
        raise ValueError("Labels must be between 0 and 9.")

    probabilities = model.predict(
        x_train,
        batch_size=batch_size,
        verbose=0
    )

    if probabilities.shape != (len(x_train), 10):
        raise ValueError("Model must return 10-class probabilities.")

    probabilities = np.clip(probabilities, 1e-7, 1.0)

    losses = -np.log(
        probabilities[np.arange(len(y_train)), y_train]
    )

    risk_scores = calculate_risk_scores(losses)

    if detection_rate is not None:
        # Select exactly the requested number of highest-risk samples.
        count = max(1, int(np.ceil(len(x_train) * detection_rate / 100)))

        suspicious_indices = np.argsort(losses)[-count:]
        suspicious_indices = np.sort(suspicious_indices)

    else:
        suspicious_indices = np.where(
            risk_scores >= threshold
        )[0]

    predictions = np.argmax(probabilities, axis=1)

    return {
        "suspicious_indices": suspicious_indices,
        "risk_scores": risk_scores,
        "losses": losses,
        "actual_labels": y_train,
        "predicted_labels": predictions,
        "suspicious_count": len(suspicious_indices),
        "detection_rate": (
            len(suspicious_indices) / len(x_train) * 100
        )
    }
