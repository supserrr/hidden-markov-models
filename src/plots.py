"""Figures for the notebook and the report."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import ACTIVITIES

COLORS = {
    "standing": "#4C78A8",
    "walking": "#F58518",
    "jumping": "#E45756",
    "still": "#54A24B",
}


def _style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )


def plot_raw_magnitude(examples):
    """`examples` maps an activity name to (time, magnitude) arrays.

    The four panels share a vertical scale so a jump reads as larger than
    sway, rather than each panel stretching its own noise to fill the axes.
    """
    _style()
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 6.4), sharex=True, sharey=True)
    for ax, name in zip(axes, ACTIVITIES):
        time, magnitude = examples[name]
        ax.plot(time, magnitude, color=COLORS[name], linewidth=1.0)
        ax.set_ylabel("m/s²")
        ax.set_title(name.capitalize(), loc="left", fontsize=11)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Accelerometer magnitude, one held-out file per activity", fontsize=13)
    fig.tight_layout()
    return fig


def plot_likelihood(history):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(np.arange(1, len(history) + 1), history, color="#4C78A8", marker="o")
    ax.set_xlabel("Baum-Welch iteration")
    ax.set_ylabel("Training log-likelihood")
    ax.set_title("Convergence of EM")
    fig.tight_layout()
    return fig


def plot_heatmap(matrix, names, title, xlabel, ylabel, fmt):
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), [name.capitalize() for name in names])
    ax.set_yticks(range(len(names)), [name.capitalize() for name in names])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > np.max(matrix) * 0.65 else "#1a1a1a"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_decoded(time, truth, predicted, names):
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.step(time, truth, where="mid", color="#4C78A8", linewidth=2.0, label="Recorded activity")
    ax.step(
        time,
        predicted + 0.08,
        where="mid",
        color="#E45756",
        linewidth=1.4,
        linestyle="--",
        label="Viterbi",
    )
    ax.set_yticks(range(len(names)), [name.capitalize() for name in names])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")
    ax.set_title("Decoded activity on one held-out mixed recording")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig
