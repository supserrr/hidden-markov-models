"""Turn Sensor Logger exports into project CSVs and rebuild the manifest.

Each zip under RECORDINGS/<Activity>/ is one recording. On iOS,
Accelerometer.csv is user acceleration with gravity already removed, in
m/s^2, and Gyroscope.csv is rad/s. Both carry `seconds_elapsed`, which
becomes `timestamp`. The converter owns data/raw: every run rewrites the
CSVs there from the zips.

Run from the repository root:

    python -m src.convert
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

from src.data import read_manifest, repo_root, write_recording
from src.features import ACTIVITIES

MANIFEST_COLUMNS = [
    "path",
    "activity",
    "split",
    "source",
    "sample_rate_hz",
    "duration_s",
    "n_rows",
    "origin",
]

# The last recordings of each activity are held out. They were taken later in
# the session than every training file of that activity, so no test window
# sits between two training windows in time.
TEST_PER_ACTIVITY = 10


def _read_axes(archive, name):
    with archive.open(name) as handle:
        rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    if not rows:
        raise ValueError(f"{name} is empty")
    seconds = np.array([float(row["seconds_elapsed"]) for row in rows])
    # Sensor Logger writes the axes as z, y, x. Read them by name.
    values = np.array([[float(row[axis]) for axis in ("x", "y", "z")] for row in rows])
    return seconds, values


def read_export(path):
    """Accelerometer and gyroscope on the accelerometer clock, starting at 0 s."""
    with zipfile.ZipFile(path) as archive:
        acc_t, acc = _read_axes(archive, "Accelerometer.csv")
        gyro_t, gyro = _read_axes(archive, "Gyroscope.csv")
    # Keep the span both sensors cover, then put the gyroscope on the
    # accelerometer timestamps. Sensor Logger usually samples the two
    # together, so this is a copy rather than an estimate.
    keep = (acc_t >= gyro_t[0]) & (acc_t <= gyro_t[-1])
    acc_t, acc = acc_t[keep], acc[keep]
    gyro = np.column_stack([np.interp(acc_t, gyro_t, gyro[:, i]) for i in range(3)])
    return acc_t - acc_t[0], acc, gyro


def activity_folders(recordings):
    """Map each activity to its folder, ignoring case and trailing spaces."""
    found = {}
    for folder in sorted(Path(recordings).iterdir()):
        name = folder.name.strip().lower()
        if folder.is_dir() and name in ACTIVITIES:
            found[name] = folder
    return found


def convert(recordings=None, root=None, test_per_activity=TEST_PER_ACTIVITY):
    root = Path(root) if root else repo_root()
    recordings = Path(recordings) if recordings else root / "RECORDINGS"
    folders = activity_folders(recordings)
    missing = [name for name in ACTIVITIES if name not in folders]
    if missing:
        raise FileNotFoundError(f"no recordings folder for {missing} under {recordings}")

    rows = []
    for name in ACTIVITIES:
        # Sensor Logger names each zip by its start time, so name order is time order.
        exports = sorted(folders[name].glob("*.zip"))
        out_dir = root / "data" / "raw" / name
        for stale in out_dir.glob(f"{name}_*.csv"):
            stale.unlink()
        first_test = len(exports) - test_per_activity
        for index, export in enumerate(exports):
            timestamps, acc, gyro = read_export(export)
            relative = Path("data") / "raw" / name / f"{name}_{index + 1:03d}.csv"
            write_recording(root / relative, timestamps, acc, gyro, [name] * len(timestamps))
            rows.append(
                {
                    "path": relative.as_posix(),
                    "activity": name,
                    "split": "test" if index >= first_test else "train",
                    "source": "sensor-logger",
                    "sample_rate_hz": f"{1.0 / np.median(np.diff(timestamps)):.0f}",
                    "duration_s": f"{timestamps[-1]:.2f}",
                    "n_rows": len(timestamps),
                    "origin": f"{export.parent.name.strip()}/{export.name}",
                }
            )

    # Mixed sequences are labeled by hand in data/sequences, so their rows stay.
    for row in read_manifest(root):
        if row.get("activity") != "mixed":
            continue
        if str(row.get("source", "")).strip().lower() == "placeholder":
            continue
        rows.append(row)

    with open(root / "data" / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _print_summary(rows):
    for name in ACTIVITIES + ["mixed"]:
        mine = [row for row in rows if row["activity"] == name]
        if not mine:
            continue
        lengths = [float(row["duration_s"] or 0) for row in mine]
        seconds = sum(lengths)
        test = sum(1 for row in mine if row["split"] == "test")
        print(
            f"{name:9} {len(mine):3d} files  {seconds:6.1f} s  "
            f"{min(lengths):.2f}-{max(lengths):.2f} s each  {test} test"
        )


if __name__ == "__main__":
    _print_summary(convert(sys.argv[1] if len(sys.argv) > 1 else None))
