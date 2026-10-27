"""Figures for the notebook and the report."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import ACTIVITIES, LOG_FEATURES

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
    Gravity is already out of the signal, and a desk and a landing differ by
    about four orders of magnitude, so the shared scale is logarithmic.
    """
    _style()
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 6.4), sharex=True, sharey=True)
    for ax, name in zip(axes, ACTIVITIES):
        time, magnitude = examples[name]
        ax.plot(time, np.maximum(magnitude, 1e-4), color=COLORS[name], linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("m/s²")
        ax.set_title(name.capitalize(), loc="left", fontsize=11)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Accelerometer magnitude without gravity, one held-out file per activity",
        fontsize=13,
    )
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


def plot_emissions(mu, var, names, features):
    """Mean of each state's Gaussian on every feature, in scaled units.

    Each cell also prints the standard deviation, so the figure carries both
    parameters of the diagonal emission model. A log feature is marked so a
    reader does not compare it with the raw value.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    limit = float(np.max(np.abs(mu)))
    image = ax.imshow(mu, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    labels = [f"log {name}" if name in LOG_FEATURES else name for name in features]
    ax.set_xticks(range(len(features)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), [name.capitalize() for name in names])
    ax.set_ylabel("Hidden state")
    ax.set_title("Emission Gaussians: mean ± standard deviation (z-score units)")
    sd = np.sqrt(var)
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            color = "white" if abs(mu[i, j]) > limit * 0.6 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{mu[i, j]:.2f}\n±{sd[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Mean")
    fig.tight_layout()
    return fig


def plot_decoded(
    time, truth, predicted, names, title="Decoded activity on one held-out mixed recording"
):
    _style()
    # Transition windows (-1) have no recorded activity, so the line breaks there.
    truth = np.where(np.asarray(truth) < 0, np.nan, truth)
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
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    return fig
