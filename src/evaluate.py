"""Confusion matrix and one-vs-rest sensitivity / specificity."""

from __future__ import annotations

import numpy as np


def confusion_matrix(truth, predicted, n_states):
    matrix = np.zeros((n_states, n_states), dtype=int)
    for true_state, guess in zip(truth, predicted):
        matrix[int(true_state), int(guess)] += 1
    return matrix


def per_class_metrics(matrix):
    """Sensitivity and specificity from a matrix with true labels on the rows."""
    total = int(matrix.sum())
    rows = []
    correct = int(np.trace(matrix))
    overall = correct / total if total else 0.0
    for state in range(matrix.shape[0]):
        tp = int(matrix[state, state])
        support = int(matrix[state].sum())
        predicted = int(matrix[:, state].sum())
        fp = predicted - tp
        fn = support - tp
        tn = total - tp - fp - fn
        sensitivity = tp / support if support else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "support": support,
                "tp": tp,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "overall_accuracy": overall,
            }
        )
    return rows


def majority_label(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[int(np.argmax(counts))]
