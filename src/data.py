"""Read project CSV recordings and the manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COLUMNS = [
    "timestamp",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "activity",
]


def repo_root():
    return Path(__file__).resolve().parents[1]


def read_recording(path):
    """Load one CSV with the project header.

    `timestamp` is seconds. Acceleration is m/s^2. Gyroscope is rad/s.
    `activity` is one of standing, walking, jumping, still, and may change
    inside a mixed sequence.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty recording: {path}")
    missing = [name for name in COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    timestamps = np.array([float(row["timestamp"]) for row in rows])
    acc = np.array(
        [[float(row[name]) for name in ("acc_x", "acc_y", "acc_z")] for row in rows]
    )
    gyro = np.array(
        [[float(row[name]) for name in ("gyro_x", "gyro_y", "gyro_z")] for row in rows]
    )
    activities = [row["activity"] for row in rows]
    return timestamps, acc, gyro, activities


def read_manifest(root=None):
    root = Path(root) if root else repo_root()
    path = root / "data" / "manifest.csv"
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_recording(path, timestamps, acc, gyro, activities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i in range(len(timestamps)):
            writer.writerow(
                [
                    f"{timestamps[i]:.6f}",
                    f"{acc[i, 0]:.6f}",
                    f"{acc[i, 1]:.6f}",
                    f"{acc[i, 2]:.6f}",
                    f"{gyro[i, 0]:.6f}",
                    f"{gyro[i, 1]:.6f}",
                    f"{gyro[i, 2]:.6f}",
                    activities[i],
                ]
            )
