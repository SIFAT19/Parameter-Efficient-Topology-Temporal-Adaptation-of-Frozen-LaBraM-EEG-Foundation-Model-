"""
metrics.py

Utility functions for evaluating binary classification experiments.


"""

import os
import csv
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_prob=None, num_classes=None):
    """
    Calculate evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    y_prob : array-like or None
        Predicted probabilities for class 1.
        Needed only for ROC-AUC.

    num_classes : int or None
        Number of classes for explicit confusion matrix labels.

    Returns
    -------
    dict
    """

    results = {}

    results["Accuracy"] = accuracy_score(y_true, y_pred)
    results["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)

    unique_labels = np.unique(y_true)
    is_binary = unique_labels.size <= 2

    if is_binary:
        labels = [0, 1]
        results["Precision"] = precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            zero_division=0,
        )
        results["Recall"] = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            zero_division=0,
        )
        results["F1"] = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            zero_division=0,
        )
    else:
        results["Precision"] = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        results["Recall"] = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        results["F1"] = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

    if y_prob is not None:
        # roc_auc_score requires both classes present in y_true
        if unique_labels.size > 1:
            try:
                results["ROC-AUC"] = roc_auc_score(y_true, y_prob)
            except Exception:
                results["ROC-AUC"] = np.nan
        else:
            results["ROC-AUC"] = np.nan
    else:
        results["ROC-AUC"] = np.nan

    # lowercase aliases for consistency with downstream code
    results["accuracy"] = results["Accuracy"]
    results["balanced_accuracy"] = results["Balanced Accuracy"]
    results["precision"] = results["Precision"]
    results["recall"] = results["Recall"]
    results["f1"] = results["F1"]
    results["roc_auc"] = results["ROC-AUC"]

    # specify labels to silence sklearn warning when only a single label appears
    if num_classes is None:
        labels = [0, 1] if is_binary else list(range(unique_labels.max() + 1))
    else:
        labels = list(range(num_classes))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return results, cm


def save_metrics(metrics, save_dir):
    """
    Save metrics.csv
    """

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "metrics.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        for key, value in metrics.items():
            writer.writerow([key, value])


def save_confusion_matrix(cm, save_dir):
    """
    Save confusion_matrix.csv
    """

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "confusion_matrix.csv")

    np.savetxt(
        file_path,
        cm,
        delimiter=",",
        fmt="%d"
    )


def save_predictions(y_true, y_pred, y_prob, save_dir):
    """
    Save predictions.csv
    """

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "predictions.csv")

    with open(file_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "True Label",
            "Predicted Label",
            "Probability"
        ])

        if y_prob is None:
            for t, p in zip(y_true, y_pred):
                writer.writerow([t, p, ""])
        else:
            for t, p, prob in zip(y_true, y_pred, y_prob):
                writer.writerow([t, p, prob])


def append_results_csv(metrics,
                       task,
                       subject,
                       epoch,
                       output_file="results/results.csv"):
    """
    Append one experiment to the master results.csv
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    file_exists = os.path.exists(output_file)

    with open(output_file, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Task",
                "Subject",
                "Epoch",
                "Accuracy",
                "Balanced Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC"
            ])

        writer.writerow([
            task,
            subject,
            epoch,
            metrics["Accuracy"],
            metrics["Balanced Accuracy"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1"],
            metrics["ROC-AUC"]
        ])