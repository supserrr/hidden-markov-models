"""Train the HMM and write metrics plus figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import read_recording, recording_rows, repo_root
from src.evaluate import confusion_matrix, majority_label, per_class_metrics
from src.features import (
    ACTIVITIES,
    HOP,
    TARGET_FS,
    WINDOW,
    ZScore,
    frames_from_resampled,
    resample,
)
from src.hmm import GaussianHMM, self_test
from src.plots import plot_decoded, plot_heatmap, plot_likelihood, plot_raw_magnitude


def _sequence(path):
    timestamps, acc, gyro, activities = read_recording(path)
    grid, acc_r, gyro_r = resample(timestamps, acc, gyro, TARGET_FS)
    # Nearest original label onto the uniform grid.
    index = np.searchsorted(timestamps, grid, side="left")
    index = np.clip(index, 0, len(activities) - 1)
    labels = [activities[int(i)] for i in index]
    features, frame_labels, centers = frames_from_resampled(acc_r, gyro_r, labels)
    coded = np.array([ACTIVITIES.index(label) for label in frame_labels], dtype=int)
    magnitude = np.linalg.norm(acc_r, axis=1)
    duration = float(grid[-1] - grid[0])
    return {
        "features": features,
        "labels": coded,
        "centers": centers,
        "time": grid - grid[0],
        "magnitude": magnitude,
        "duration": duration,
    }


def load_splits(root=None):
    root = Path(root) if root else repo_root()
    groups = {"train": [], "test": []}
    rows = recording_rows(root)
    if not rows:
        raise FileNotFoundError(
            "No recordings yet. Replace the placeholder files listed in data/manifest.csv."
        )
    for row in rows:
        item = _sequence(root / row["path"])
        item["meta"] = row
        groups[row["split"]].append(item)
    return groups


def fit_model(splits):
    train = splits["train"]
    raw = np.concatenate([item["features"] for item in train], axis=0)
    scaler = ZScore().fit(raw)
    sequences = [scaler.transform(item["features"]) for item in train]
    labels = [item["labels"] for item in train]
    model = GaussianHMM(n_states=len(ACTIVITIES), n_features=raw.shape[1])
    model.initialize_from_labels(sequences, labels, prior=1.0)
    supervised = _clone_params(model)
    history = model.fit(sequences, max_iter=40, tol=1e-6)
    order = model.align_to_labels(sequences, labels)
    return model, scaler, history, order, supervised


def _clone_params(model):
    return {
        "pi": model.pi.copy(),
        "A": model.A.copy(),
        "mu": model.mu.copy(),
        "var": model.var.copy(),
    }


def _score_items(model, scaler, items):
    truth = []
    predicted = []
    file_hits = []
    decoded_example = None
    for item in items:
        features = scaler.transform(item["features"])
        path = model.viterbi(features)
        truth.append(item["labels"])
        predicted.append(path)
        file_hits.append(int(majority_label(path) == majority_label(item["labels"])))
        if decoded_example is None and item["meta"]["activity"] == "mixed":
            decoded_example = {
                "centers": item["centers"],
                "truth": item["labels"],
                "predicted": path,
                "path": item["meta"]["path"],
            }
    truth = np.concatenate(truth)
    predicted = np.concatenate(predicted)
    matrix = confusion_matrix(truth, predicted, len(ACTIVITIES))
    metrics = per_class_metrics(matrix)
    return {
        "confusion": matrix,
        "metrics": metrics,
        "frame_accuracy": float(np.mean(truth == predicted)),
        "file_accuracy": float(np.mean(file_hits)) if file_hits else 0.0,
        "n_frames": int(len(truth)),
        "n_files": len(items),
        "decoded_example": decoded_example,
    }


def _score_with_params(model, scaler, items, params):
    """Score a saved parameter set, then put the fitted parameters back."""
    current = _clone_params(model)
    model.pi = params["pi"]
    model.A = params["A"]
    model.mu = params["mu"]
    model.var = params["var"]
    try:
        return _score_items(model, scaler, items)
    finally:
        model.pi = current["pi"]
        model.A = current["A"]
        model.mu = current["mu"]
        model.var = current["var"]


def class_feature_means(splits, scaler):
    sums = {name: [] for name in ACTIVITIES}
    for item in splits["train"]:
        scaled = scaler.transform(item["features"])
        for state, name in enumerate(ACTIVITIES):
            taken = scaled[item["labels"] == state]
            if len(taken):
                sums[name].append(taken)
    return {
        name: np.concatenate(rows).mean(axis=0).tolist() if rows else []
        for name, rows in sums.items()
    }


def run(root=None):
    root = Path(root) if root else repo_root()
    check = self_test()
    splits = load_splits(root)
    model, scaler, history, order, supervised = fit_model(splits)
    pure_test = [item for item in splits["test"] if item["meta"]["activity"] != "mixed"]
    mixed_test = [item for item in splits["test"] if item["meta"]["activity"] == "mixed"]
    held_out = _score_items(model, scaler, pure_test)
    mixed = _score_items(model, scaler, mixed_test)
    supervised_score = _score_with_params(model, scaler, pure_test, supervised)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    examples = {}
    for name in ACTIVITIES:
        candidates = [item for item in pure_test if item["meta"]["activity"] == name]
        match = max(candidates, key=lambda item: item["duration"])
        # Same eight-second span on every panel, on one shared vertical scale.
        mask = match["time"] <= 8.0
        examples[name] = (match["time"][mask], match["magnitude"][mask])
    plot_raw_magnitude(examples).savefig(figures / "raw_magnitude.png", dpi=150)
    plot_likelihood(history).savefig(figures / "likelihood.png", dpi=150)
    plot_heatmap(
        model.A,
        ACTIVITIES,
        "Transition matrix",
        "To",
        "From",
        ".2f",
    ).savefig(figures / "transition.png", dpi=150)
    plot_heatmap(
        held_out["confusion"],
        ACTIVITIES,
        "Confusion matrix on held-out single-activity files",
        "Predicted",
        "Recorded",
        "d",
    ).savefig(figures / "confusion.png", dpi=150)
    example = mixed["decoded_example"]
    plot_decoded(
        example["centers"],
        example["truth"],
        example["predicted"],
        ACTIVITIES,
    ).savefig(figures / "decoded_sequence.png", dpi=150)

    import matplotlib.pyplot as plt

    plt.close("all")

    rates = {}
    durations = {name: 0.0 for name in ACTIVITIES}
    file_counts = {name: 0 for name in ACTIVITIES}
    for row in recording_rows(root):
        rates[row["sample_rate_hz"]] = rates.get(row["sample_rate_hz"], 0) + 1
        if row["activity"] in durations:
            durations[row["activity"]] += float(row["duration_s"])
            file_counts[row["activity"]] += 1

    payload = {
        "self_test": check,
        "state_order_after_alignment": [ACTIVITIES[i] for i in order.tolist()],
        "alignment_is_identity": bool(np.array_equal(order, np.arange(len(ACTIVITIES)))),
        "iterations": len(history),
        "loglik_start": history[0],
        "loglik_end": history[-1],
        "loglik_history": history,
        "transition": model.A.tolist(),
        "pi": model.pi.tolist(),
        "held_out_frames": held_out["metrics"],
        "held_out_confusion": held_out["confusion"].tolist(),
        "held_out_frame_accuracy": held_out["frame_accuracy"],
        "held_out_file_accuracy": held_out["file_accuracy"],
        "held_out_n_frames": held_out["n_frames"],
        "held_out_n_files": held_out["n_files"],
        "supervised_frame_accuracy": supervised_score["frame_accuracy"],
        "mixed_frame_accuracy": mixed["frame_accuracy"],
        "mixed_confusion": mixed["confusion"].tolist(),
        "mixed_n_frames": mixed["n_frames"],
        "mixed_n_files": mixed["n_files"],
        "decoded_example": example["path"],
        "feature_means_z": class_feature_means(splits, scaler),
        "sample_rates": rates,
        "duration_s": durations,
        "files_per_activity": file_counts,
        "window_s": WINDOW / TARGET_FS,
        "hop_s": HOP / TARGET_FS,
        "target_fs": TARGET_FS,
    }
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    _print_summary(payload)
    return payload


def _print_summary(payload):
    print(
        f"self-test viterbi={payload['self_test']['viterbi_accuracy']:.3f}  "
        f"EM {payload['loglik_start']:.1f} -> {payload['loglik_end']:.1f} "
        f"in {payload['iterations']} iters  "
        f"identity_align={payload['alignment_is_identity']}"
    )
    print(
        f"held-out frame accuracy={payload['held_out_frame_accuracy']:.3f}  "
        f"file accuracy={payload['held_out_file_accuracy']:.3f}  "
        f"supervised={payload['supervised_frame_accuracy']:.3f}  "
        f"mixed={payload['mixed_frame_accuracy']:.3f}"
    )
    for name, row in zip(ACTIVITIES, payload["held_out_frames"]):
        print(
            f"  {name:10} n={row['support']:4d}  "
            f"sens={row['sensitivity']:.3f}  spec={row['specificity']:.3f}"
        )


if __name__ == "__main__":
    run()
