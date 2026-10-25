"""Write simulated accelerometer and gyroscope CSV files.

The files follow the column layout the loader expects and are long enough
for the windowing used in the notebook. They are not phone recordings.
`source` in data/manifest.csv is `synthetic` on every row.

Rates are mixed on purpose (40, 50, and 100 Hz) so resampling is exercised.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data import repo_root, write_recording
from src.features import ACTIVITIES

SEED = 7
FILES_PER_ACTIVITY = 50
TEST_PER_ACTIVITY = 10
RATES = (40.0, 50.0, 100.0)

MIXED_ORDERS = [
    ["still", "standing", "walking", "standing"],
    ["standing", "walking", "jumping", "walking"],
    ["still", "standing", "walking", "jumping"],
    ["walking", "standing", "still", "standing"],
    ["standing", "walking", "standing", "still"],
    ["walking", "jumping", "walking", "standing"],
]


def _imu(activity, duration, fs, rng):
    count = int(round(duration * fs))
    time = np.arange(count) / fs
    acc = np.zeros((count, 3))
    gyro = np.zeros((count, 3))
    # Phone held upright: gravity on +z, with a small fixed tilt per file.
    tilt_x = float(rng.normal(0.0, 0.35))
    tilt_y = float(rng.normal(0.0, 0.35))
    acc[:, 0] = tilt_x
    acc[:, 1] = tilt_y
    acc[:, 2] = 9.81

    if activity == "still":
        acc += rng.normal(0.0, 0.02, size=acc.shape)
        gyro += rng.normal(0.0, 0.008, size=gyro.shape)
    elif activity == "standing":
        sway = float(rng.uniform(0.18, 0.4))
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(0.08, 0.22))
        acc[:, 0] += amp * np.sin(2 * np.pi * sway * time + phase)
        acc[:, 1] += amp * 0.7 * np.sin(2 * np.pi * sway * time + phase + 0.6)
        acc[:, 2] += amp * 0.25 * np.sin(2 * np.pi * sway * 2 * time)
        gyro[:, 0] += 0.12 * amp * np.sin(2 * np.pi * sway * time)
        gyro[:, 1] += 0.10 * amp * np.sin(2 * np.pi * sway * time + 1.0)
        acc += rng.normal(0.0, 0.04, size=acc.shape)
        gyro += rng.normal(0.0, 0.02, size=gyro.shape)
    elif activity == "walking":
        step = float(rng.uniform(1.6, 2.05))
        phase = float(rng.uniform(0, 2 * np.pi))
        vertical = float(rng.uniform(1.6, 2.8))
        acc[:, 2] += vertical * np.sin(2 * np.pi * step * time + phase)
        acc[:, 2] += 0.45 * vertical * np.sin(2 * np.pi * 2 * step * time + phase)
        acc[:, 0] += float(rng.uniform(0.7, 1.3)) * np.sin(2 * np.pi * step * time + phase + 1.1)
        acc[:, 1] += float(rng.uniform(0.45, 0.9)) * np.sin(2 * np.pi * (step / 2) * time)
        gyro[:, 0] += float(rng.uniform(0.55, 1.05)) * np.sin(2 * np.pi * step * time)
        gyro[:, 1] += float(rng.uniform(0.35, 0.75)) * np.sin(2 * np.pi * step * time + 0.4)
        gyro[:, 2] += float(rng.uniform(0.15, 0.4)) * np.sin(2 * np.pi * (step / 2) * time)
        acc += rng.normal(0.0, 0.18, size=acc.shape)
        gyro += rng.normal(0.0, 0.08, size=gyro.shape)
    elif activity == "jumping":
        rate = float(rng.uniform(1.45, 1.9))
        first = float(rng.uniform(0.15, 0.45))
        landings = np.arange(first, time[-1] - 0.05, 1.0 / rate)
        for landing in landings:
            flight = np.exp(-0.5 * ((time - (landing - 0.12)) / 0.035) ** 2)
            impact = np.exp(-0.5 * ((time - landing) / 0.022) ** 2)
            acc[:, 2] += -float(rng.uniform(2.2, 3.6)) * flight
            acc[:, 2] += float(rng.uniform(11.0, 16.0)) * impact
            acc[:, 0] += float(rng.uniform(-1.2, 1.2)) * impact
            acc[:, 1] += float(rng.uniform(-1.2, 1.2)) * impact
            sign = float(rng.choice([-1.0, 1.0]))
            gyro[:, 0] += sign * float(rng.uniform(1.8, 3.2)) * impact
            gyro[:, 1] += -sign * float(rng.uniform(1.0, 2.2)) * impact
        acc += rng.normal(0.0, 0.22, size=acc.shape)
        gyro += rng.normal(0.0, 0.1, size=gyro.shape)
    else:
        raise ValueError(activity)
    activities = [activity] * count
    return time, acc, gyro, activities


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "activity",
        "split",
        "source",
        "sample_rate_hz",
        "duration_s",
        "n_rows",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(root=None, seed=SEED):
    root = Path(root) if root else repo_root()
    rng = np.random.default_rng(seed)
    manifest = []

    for activity in ACTIVITIES:
        test_ids = set(rng.choice(FILES_PER_ACTIVITY, size=TEST_PER_ACTIVITY, replace=False).tolist())
        for index in range(FILES_PER_ACTIVITY):
            split = "test" if index in test_ids else "train"
            duration = float(rng.uniform(6.5, 9.5))
            fs = float(RATES[int(rng.integers(0, len(RATES)))])
            time, acc, gyro, activities = _imu(activity, duration, fs, rng)
            relative = Path("data") / "raw" / activity / f"{activity}_{index + 1:03d}.csv"
            write_recording(root / relative, time, acc, gyro, activities)
            manifest.append(
                {
                    "path": relative.as_posix(),
                    "activity": activity,
                    "split": split,
                    "source": "synthetic",
                    "sample_rate_hz": f"{fs:.0f}",
                    "duration_s": f"{duration:.3f}",
                    "n_rows": str(len(time)),
                }
            )

    n_mixed = 16
    test_mixed = set(rng.choice(n_mixed, size=4, replace=False).tolist())
    for index in range(n_mixed):
        split = "test" if index in test_mixed else "train"
        order = MIXED_ORDERS[index % len(MIXED_ORDERS)]
        fs = float(RATES[int(rng.integers(0, len(RATES)))])
        pieces_t = []
        pieces_acc = []
        pieces_gyro = []
        pieces_act = []
        cursor = 0.0
        for activity in order:
            duration = float(rng.uniform(6.5, 8.5))
            time, acc, gyro, activities = _imu(activity, duration, fs, rng)
            pieces_t.append(time + cursor)
            pieces_acc.append(acc)
            pieces_gyro.append(gyro)
            pieces_act.extend(activities)
            cursor += duration
        time = np.concatenate(pieces_t)
        acc = np.concatenate(pieces_acc)
        gyro = np.concatenate(pieces_gyro)
        relative = Path("data") / "sequences" / f"sequence_{index + 1:03d}.csv"
        write_recording(root / relative, time, acc, gyro, pieces_act)
        manifest.append(
            {
                "path": relative.as_posix(),
                "activity": "mixed",
                "split": split,
                "source": "synthetic",
                "sample_rate_hz": f"{fs:.0f}",
                "duration_s": f"{cursor:.3f}",
                "n_rows": str(len(time)),
            }
        )

    _write_manifest(root / "data" / "manifest.csv", manifest)
    return manifest


if __name__ == "__main__":
    rows = generate()
    pure = [row for row in rows if row["activity"] != "mixed"]
    print(f"wrote {len(rows)} files ({len(pure)} single-activity, {len(rows) - len(pure)} mixed)")
