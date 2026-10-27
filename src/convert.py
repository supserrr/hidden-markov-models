"""Turn Sensor Logger exports into project CSVs and rebuild the manifest.

Each zip under data/raw/<activity>/ is one recording, exactly as Sensor
Logger exported it. On iOS, Accelerometer.csv is user acceleration with
gravity already removed, in m/s^2, and Gyroscope.csv is rad/s. Both carry
`seconds_elapsed`, which becomes `timestamp`. The converter owns
data/processed: every run rewrites the CSVs there from the zips.

Recordings from a later session go under data/unseen/raw/ instead, in the
same activity folders plus `mixed/` for continuous recordings that change
activity. Their CSVs land in data/unseen/processed/ with split `unseen`, so
they never enter training or the same-session test set. Each mixed
recording is labelled from the spans in data/unseen/segments.csv.

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
from src.features import ACTIVITIES, TRANSITION

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

UNSEEN = Path("data") / "unseen"


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


def _relative(path, root):
    """Repository-relative path when the export lives inside the repository."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def activity_folders(recordings):
    """Map each activity to its folder, ignoring case and trailing spaces."""
    found = {}
    for folder in sorted(Path(recordings).iterdir()):
        name = folder.name.strip().lower()
        if folder.is_dir() and name in ACTIVITIES:
            found[name] = folder
    return found


def _manifest_row(relative, activity, split, timestamps, export, root):
    return {
        "path": relative.as_posix(),
        "activity": activity,
        "split": split,
        "source": "sensor-logger",
        "sample_rate_hz": f"{1.0 / np.median(np.diff(timestamps)):.0f}",
        "duration_s": f"{timestamps[-1] - timestamps[0]:.2f}",
        "n_rows": len(timestamps),
        "origin": _relative(export, root),
    }


def read_segments(path):
    """Activity spans for each mixed recording, keyed by zip name.

    Each row of segments.csv is `recording,start_s,end_s,activity`, with
    times in seconds from the start of that recording.
    """
    spans = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            spans.setdefault(row["recording"].strip(), []).append(
                (float(row["start_s"]), float(row["end_s"]), row["activity"].strip().lower())
            )
    return {name: sorted(rows) for name, rows in spans.items()}


def label_rows(timestamps, spans):
    """A label for every row inside the spans, and a mask of those rows.

    The spans must meet end to end, so no row in the middle goes unlabelled.
    Rows before the first span or after the last are dropped.
    """
    for (_, end, _), (start, _, _) in zip(spans[:-1], spans[1:]):
        if abs(end - start) > 1e-6:
            raise ValueError(f"segments leave a gap or overlap at {end} s")
    for _, _, activity in spans:
        if activity not in ACTIVITIES and activity != TRANSITION:
            raise ValueError(f"unknown activity {activity!r} in segments")
    keep = (timestamps >= spans[0][0]) & (timestamps < spans[-1][1])
    starts = np.array([start for start, _, _ in spans])
    which = np.searchsorted(starts, timestamps[keep], side="right") - 1
    return keep, [spans[i][2] for i in which]


def convert_unseen(root):
    """Write data/unseen/processed from data/unseen/raw and return its manifest rows."""
    raw = root / UNSEEN / "raw"
    if not raw.is_dir():
        return []
    rows = []
    folders = activity_folders(raw)
    for name in ACTIVITIES:
        if name not in folders:
            continue
        out_dir = root / UNSEEN / "processed" / name
        for stale in out_dir.glob(f"{name}_*.csv"):
            stale.unlink()
        for index, export in enumerate(sorted(folders[name].glob("*.zip"))):
            timestamps, acc, gyro = read_export(export)
            relative = UNSEEN / "processed" / name / f"{name}_{index + 1:03d}.csv"
            write_recording(root / relative, timestamps, acc, gyro, [name] * len(timestamps))
            rows.append(_manifest_row(relative, name, "unseen", timestamps, export, root))

    exports = sorted((raw / "mixed").glob("*.zip"))
    for stale in (root / UNSEEN / "processed" / "mixed").glob("mixed_*.csv"):
        stale.unlink()
    segments = read_segments(root / UNSEEN / "segments.csv") if exports else {}
    for index, export in enumerate(exports):
        if export.name not in segments:
            raise ValueError(f"{export.name} has no rows in {UNSEEN / 'segments.csv'}")
        timestamps, acc, gyro = read_export(export)
        keep, labels = label_rows(timestamps, segments[export.name])
        timestamps, acc, gyro = timestamps[keep], acc[keep], gyro[keep]
        relative = UNSEEN / "processed" / "mixed" / f"mixed_{index + 1:03d}.csv"
        write_recording(root / relative, timestamps, acc, gyro, labels)
        rows.append(_manifest_row(relative, "mixed", "unseen", timestamps, export, root))
    return rows


def convert(recordings=None, root=None, test_per_activity=TEST_PER_ACTIVITY):
    root = Path(root) if root else repo_root()
    recordings = Path(recordings) if recordings else root / "data" / "raw"
    folders = activity_folders(recordings)
    missing = [name for name in ACTIVITIES if name not in folders]
    if missing:
        raise FileNotFoundError(f"no recordings folder for {missing} under {recordings}")

    rows = []
    for name in ACTIVITIES:
        # Sensor Logger names each zip by its start time, so name order is time order.
        exports = sorted(folders[name].glob("*.zip"))
        out_dir = root / "data" / "processed" / name
        for stale in out_dir.glob(f"{name}_*.csv"):
            stale.unlink()
        first_test = len(exports) - test_per_activity
        for index, export in enumerate(exports):
            timestamps, acc, gyro = read_export(export)
            relative = Path("data") / "processed" / name / f"{name}_{index + 1:03d}.csv"
            write_recording(root / relative, timestamps, acc, gyro, [name] * len(timestamps))
            split = "test" if index >= first_test else "train"
            rows.append(_manifest_row(relative, name, split, timestamps, export, root))

    rows.extend(convert_unseen(root))

    # Mixed sequences in data/sequences are labeled by hand, so their rows
    # stay. The ones under data/unseen were rebuilt above.
    for row in read_manifest(root):
        if row.get("activity") != "mixed" or row["path"].startswith(UNSEEN.as_posix()):
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
        unseen = sum(1 for row in mine if row["split"] == "unseen")
        print(
            f"{name:9} {len(mine):3d} files  {seconds:6.1f} s  "
            f"{min(lengths):.2f}-{max(lengths):.2f} s each  {test} test  {unseen} unseen"
        )


if __name__ == "__main__":
    _print_summary(convert(sys.argv[1] if len(sys.argv) > 1 else None))
