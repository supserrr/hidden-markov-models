"""Train the HMM and write metrics plus figures to results/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import read_recording, recording_rows, repo_root
from src.evaluate import confusion_matrix, majority_label, per_class_metrics
from src.features import (
    ACTIVITIES,
    FEATURE_NAMES,
    HOP,
    TARGET_FS,
    WINDOW,
    FeatureScaler,
    frames_from_resampled,
    resample,
)
from src.hmm import GaussianHMM, self_test
from src.plots import (
    plot_decoded,
    plot_emissions,
    plot_heatmap,
    plot_likelihood,
    plot_raw_magnitude,
)

# Order of held-out files joined end to end when there is no mixed recording.
# It runs from rest to the most active state and back down, so the decoder has
# to make every change in both directions.
JOIN_ORDER = ["still", "standing", "walking", "jumping", "walking", "standing", "still"]


def _sequence(path):
    timestamps, acc, gyro, activities = read_recording(path)
    grid, acc_r, gyro_r = resample(timestamps, acc, gyro, TARGET_FS)
    # Nearest original label onto the uniform grid.
    index = np.searchsorted(timestamps, grid, side="left")
    index = np.clip(index, 0, len(activities) - 1)
    labels = [activities[int(i)] for i in index]
    features, frame_labels, centers = frames_from_resampled(acc_r, gyro_r, labels)
    # A transition window has no true activity, so it is coded -1 and not scored.
    coded = np.array(
        [ACTIVITIES.index(label) if label in ACTIVITIES else -1 for label in frame_labels],
        dtype=int,
    )
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
    groups = {"train": [], "test": [], "unseen": []}
    rows = recording_rows(root)
    if not rows:
        raise FileNotFoundError(
            "No recordings yet. Run `python -m src.convert` to build data/processed/ from data/raw/."
        )
    for row in rows:
        item = _sequence(root / row["path"])
        item["meta"] = row
        groups[row["split"]].append(item)
    return groups


def fit_model(splits, log=True):
    train = splits["train"]
    if any(np.any(item["labels"] < 0) for item in train):
        raise ValueError("transition spans are for scoring only; keep them out of training")
    raw = np.concatenate([item["features"] for item in train], axis=0)
    scaler = FeatureScaler(log=log).fit(raw)
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


def score_items(model, scaler, items):
    truth = []
    predicted = []
    file_hits = []
    for item in items:
        features = scaler.transform(item["features"])
        # Viterbi runs on the whole recording, transitions included, so the
        # chain sees the real timeline. Only windows with an activity count.
        path = model.viterbi(features)
        scored = item["labels"] >= 0
        truth.append(item["labels"][scored])
        predicted.append(path[scored])
        file_hits.append(int(majority_label(path[scored]) == majority_label(item["labels"][scored])))
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
    }


def score_with_params(model, scaler, items, params):
    """Score a saved parameter set, then put the fitted parameters back."""
    current = _clone_params(model)
    model.pi = params["pi"]
    model.A = params["A"]
    model.mu = params["mu"]
    model.var = params["var"]
    try:
        return score_items(model, scaler, items)
    finally:
        model.pi = current["pi"]
        model.A = current["A"]
        model.mu = current["mu"]
        model.var = current["var"]


def joined_sequence(items, order=JOIN_ORDER):
    """Held-out single-activity files placed back to back as one sequence.

    Each file is used at most once. The changes in the result are file
    boundaries, not recorded transitions, and the figure title says so.
    """
    queues = {
        name: [item for item in items if item["meta"]["activity"] == name]
        for name in ACTIVITIES
    }
    features, labels, centers, paths = [], [], [], []
    offset = 0.0
    for name in order:
        if not queues[name]:
            continue
        item = queues[name].pop(0)
        features.append(item["features"])
        labels.append(item["labels"])
        centers.append(item["centers"] + offset)
        paths.append(item["meta"]["path"])
        offset += item["duration"]
    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "centers": np.concatenate(centers),
        "paths": paths,
    }


def decoded_example(model, scaler, pure_test, mixed_test):
    """The sequence the decoded-path figure shows.

    A held-out mixed recording when one exists. Otherwise held-out
    single-activity files joined end to end.
    """
    if mixed_test:
        item = mixed_test[0]
        source = {
            "features": item["features"],
            "labels": item["labels"],
            "centers": item["centers"],
            "paths": [item["meta"]["path"]],
        }
        title = "Decoded activity on one held-out mixed recording"
    else:
        source = joined_sequence(pure_test)
        title = "Decoded activity on held-out files joined end to end"
    predicted = model.viterbi(scaler.transform(source["features"]))
    scored = source["labels"] >= 0
    return {
        "centers": source["centers"],
        "truth": source["labels"],
        "predicted": predicted,
        "paths": source["paths"],
        "title": title,
        "joined": not mixed_test,
        "accuracy": float(np.mean(predicted[scored] == source["labels"][scored])),
    }


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
    held_out = score_items(model, scaler, pure_test)
    # Mixed recordings are optional. Without one, the decoded-path figure
    # joins held-out single-activity files instead.
    mixed = score_items(model, scaler, mixed_test) if mixed_test else None
    supervised_score = score_with_params(model, scaler, pure_test, supervised)
    example = decoded_example(model, scaler, pure_test, mixed_test)
    # The unseen split is a later session. It plays no part in fitting or in
    # the held-out numbers above, and it is scored on its own.
    unseen_items = splits["unseen"]
    unseen_single = [item for item in unseen_items if item["meta"]["activity"] != "mixed"]
    unseen_mixed = [item for item in unseen_items if item["meta"]["activity"] == "mixed"]
    unseen = score_items(model, scaler, unseen_items) if unseen_items else None
    unseen_single_score = score_items(model, scaler, unseen_single) if unseen_single else None
    unseen_mixed_score = score_items(model, scaler, unseen_mixed) if unseen_mixed else None

    results = root / "results"
    figures = results / "figures"
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
        ".3f",
    ).savefig(figures / "transition.png", dpi=150)
    plot_emissions(model.mu, model.var, ACTIVITIES, FEATURE_NAMES).savefig(
        figures / "emission.png", dpi=150
    )
    plot_heatmap(
        held_out["confusion"],
        ACTIVITIES,
        "Confusion matrix on held-out single-activity files",
        "Predicted",
        "Recorded",
        "d",
    ).savefig(figures / "confusion.png", dpi=150)
    plot_decoded(
        example["centers"],
        example["truth"],
        example["predicted"],
        ACTIVITIES,
        title=example["title"],
    ).savefig(figures / "decoded_sequence.png", dpi=150)
    if unseen:
        plot_heatmap(
            unseen["confusion"],
            ACTIVITIES,
            "Confusion matrix on the unseen session",
            "Predicted",
            "Recorded",
            "d",
        ).savefig(figures / "confusion_unseen.png", dpi=150)
    unseen_decoded = []
    for number, item in enumerate(unseen_mixed, start=1):
        predicted = model.viterbi(scaler.transform(item["features"]))
        scored = item["labels"] >= 0
        unseen_decoded.append(
            {
                "path": item["meta"]["path"],
                "accuracy": float(np.mean(predicted[scored] == item["labels"][scored])),
            }
        )
        plot_decoded(
            item["centers"],
            item["labels"],
            predicted,
            ACTIVITIES,
            title=f"Decoded activity on unseen continuous recording {number}",
        ).savefig(figures / f"decoded_unseen_{number}.png", dpi=150)

    import matplotlib.pyplot as plt

    plt.close("all")

    rates = {}
    durations = {name: 0.0 for name in ACTIVITIES}
    file_counts = {name: 0 for name in ACTIVITIES}
    unseen_counts = {name: 0 for name in ACTIVITIES + ["mixed"]}
    for row in recording_rows(root):
        if row["split"] == "unseen":
            unseen_counts[row["activity"]] += 1
            continue
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
        "mixed_frame_accuracy": mixed["frame_accuracy"] if mixed else None,
        "mixed_confusion": mixed["confusion"].tolist() if mixed else None,
        "mixed_n_frames": mixed["n_frames"] if mixed else 0,
        "mixed_n_files": len(mixed_test),
        "decoded_example": example["paths"],
        "decoded_example_joined": example["joined"],
        "decoded_example_accuracy": example["accuracy"],
        "unseen_frames": unseen["metrics"] if unseen else None,
        "unseen_confusion": unseen["confusion"].tolist() if unseen else None,
        "unseen_frame_accuracy": unseen["frame_accuracy"] if unseen else None,
        "unseen_n_frames": unseen["n_frames"] if unseen else 0,
        "unseen_n_files": len(unseen_items),
        "unseen_files_per_activity": unseen_counts,
        "unseen_single_frame_accuracy": (
            unseen_single_score["frame_accuracy"] if unseen_single_score else None
        ),
        "unseen_single_file_accuracy": (
            unseen_single_score["file_accuracy"] if unseen_single_score else None
        ),
        "unseen_mixed_frame_accuracy": (
            unseen_mixed_score["frame_accuracy"] if unseen_mixed_score else None
        ),
        "unseen_mixed_n_frames": unseen_mixed_score["n_frames"] if unseen_mixed_score else 0,
        "unseen_decoded": unseen_decoded,
        "feature_means_z": class_feature_means(splits, scaler),
        "sample_rates": rates,
        "duration_s": durations,
        "files_per_activity": file_counts,
        "window_s": WINDOW / TARGET_FS,
        "hop_s": HOP / TARGET_FS,
        "target_fs": TARGET_FS,
    }
    (results / "metrics.json").write_text(json.dumps(payload, indent=2))
    _print_summary(payload)
    return payload


def _print_summary(payload):
    print(
        f"self-test viterbi={payload['self_test']['viterbi_accuracy']:.3f}  "
        f"EM {payload['loglik_start']:.1f} -> {payload['loglik_end']:.1f} "
        f"in {payload['iterations']} iters  "
        f"identity_align={payload['alignment_is_identity']}"
    )
    mixed = payload["mixed_frame_accuracy"]
    print(
        f"held-out frame accuracy={payload['held_out_frame_accuracy']:.3f}  "
        f"file accuracy={payload['held_out_file_accuracy']:.3f}  "
        f"supervised={payload['supervised_frame_accuracy']:.3f}  "
        f"mixed={'none' if mixed is None else f'{mixed:.3f}'}  "
        f"decoded example={payload['decoded_example_accuracy']:.3f}"
        f"{' (joined files)' if payload['decoded_example_joined'] else ''}"
    )
    for name, row in zip(ACTIVITIES, payload["held_out_frames"]):
        print(
            f"  {name:10} n={row['support']:4d}  "
            f"sens={row['sensitivity']:.3f}  spec={row['specificity']:.3f}"
        )
    if payload["unseen_frames"] is None:
        return
    single = payload["unseen_single_frame_accuracy"]
    mixed = payload["unseen_mixed_frame_accuracy"]
    print(
        f"unseen frame accuracy={payload['unseen_frame_accuracy']:.3f}  "
        f"single-activity={'none' if single is None else f'{single:.3f}'}  "
        f"continuous={'none' if mixed is None else f'{mixed:.3f}'}"
    )
    for name, row in zip(ACTIVITIES, payload["unseen_frames"]):
        print(
            f"  {name:10} n={row['support']:4d}  "
            f"sens={row['sensitivity']:.3f}  spec={row['specificity']:.3f}"
        )


if __name__ == "__main__":
    run()
