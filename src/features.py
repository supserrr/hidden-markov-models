"""Windowed time-domain and frequency-domain features for one IMU stream."""

from __future__ import annotations

import numpy as np

ACTIVITIES = ["standing", "walking", "jumping", "still"]

# One second at the common rate. A walk step is about half a second, so the
# window covers at least one full step and the FFT bin width is 1 Hz.
TARGET_FS = 50.0
WINDOW = 50
HOP = 25

FEATURE_NAMES = [
    "acc_mag_mean",
    "acc_mag_var",
    "acc_mag_rms",
    "sma",
    "acc_xy_corr",
    "acc_dom_freq",
    "acc_spectral_energy",
    "gyro_mag_mean",
    "gyro_dom_freq",
]

# Magnitude and energy features span about seven orders of magnitude, from a
# phone on a desk to a landing. On that scale a z-score is set by jumping, and
# standing and still end up closer together than the variance floor of one
# Gaussian, so Baum-Welch merges them. A log puts each of these features on a
# scale where a Gaussian per state fits and the quiet pair stays apart.
LOG_FEATURES = [
    "acc_mag_mean",
    "acc_mag_var",
    "acc_mag_rms",
    "sma",
    "acc_spectral_energy",
    "gyro_mag_mean",
]
LOG_FLOOR = 1e-6


def resample(timestamps, acc, gyro, fs=TARGET_FS):
    """Linear interpolation onto a uniform grid. Uses the timestamp column."""
    start = float(timestamps[0])
    stop = float(timestamps[-1])
    count = int(np.floor((stop - start) * fs)) + 1
    if count < WINDOW:
        raise ValueError("recording is shorter than one window after resampling")
    grid = start + np.arange(count) / fs
    acc_out = np.column_stack([np.interp(grid, timestamps, acc[:, i]) for i in range(3)])
    gyro_out = np.column_stack([np.interp(grid, timestamps, gyro[:, i]) for i in range(3)])
    return grid, acc_out, gyro_out


def _dominant_frequency(signal, fs):
    """Peak frequency of a window, plus band energy from 0.15 Hz to 8 Hz.

    The mean is removed first so the DC bin cannot win the argmax. The FFT
    is zero-padded to eight times the window so the peak can fall between
    the native 1 Hz bins of a one-second window. If no bin rises clearly
    above the median of the band, there is no cadence and the peak is 0 Hz.
    """
    centered = signal - np.mean(signal)
    tapered = centered * np.hanning(len(centered))
    nfft = 1 << int(np.ceil(np.log2(len(centered) * 8)))
    spectrum = np.fft.rfft(tapered, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    power = np.abs(spectrum) ** 2
    power[0] = 0.0
    band = (freqs >= 0.15) & (freqs <= 8.0)
    energy = float(power[band].sum() / len(centered))
    usable = power[band]
    if usable.size == 0 or usable.max() <= 0:
        return 0.0, energy
    median = float(np.median(usable))
    if usable.max() < 6.0 * (median + 1e-12):
        return 0.0, energy
    peak = int(np.argmax(np.where(band, power, 0.0)))
    return float(freqs[peak]), energy


def _safe_corr(left, right):
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return value


def window_features(acc, gyro, fs):
    """Nine features for one window.

    The per-window mean vector of acceleration is treated as gravity and
    removed before the magnitude features. Those features then describe
    motion rather than the tilt of the phone.
    """
    dynamic = acc - acc.mean(axis=0, keepdims=True)
    magnitude = np.linalg.norm(dynamic, axis=1)
    gyro_dynamic = gyro - gyro.mean(axis=0, keepdims=True)
    gyro_magnitude = np.linalg.norm(gyro_dynamic, axis=1)
    # Peak frequency is taken on the axis that moves the most. The magnitude
    # rectifies a back-and-forth signal, which doubles the cadence and can
    # hide it. Spectral energy stays on the magnitude so it measures total
    # motion, not the motion of one axis.
    moving = int(np.argmax(dynamic.var(axis=0)))
    peak, _ = _dominant_frequency(dynamic[:, moving], fs)
    _, energy = _dominant_frequency(magnitude, fs)
    gyro_moving = int(np.argmax(gyro_dynamic.var(axis=0)))
    gyro_peak, _ = _dominant_frequency(gyro_dynamic[:, gyro_moving], fs)
    return np.array(
        [
            float(magnitude.mean()),
            float(magnitude.var()),
            float(np.sqrt(np.mean(magnitude ** 2))),
            float(np.mean(np.abs(dynamic).sum(axis=1))),
            _safe_corr(dynamic[:, 0], dynamic[:, 1]),
            peak,
            energy,
            float(gyro_magnitude.mean()),
            gyro_peak,
        ],
        dtype=float,
    )


def frames_from_resampled(acc, gyro, activities, fs=TARGET_FS, window=WINDOW, hop=HOP):
    """Cut a resampled recording into feature rows and a majority label per row."""
    rows = []
    labels = []
    centers = []
    limit = len(acc) - window + 1
    for start in range(0, limit, hop):
        stop = start + window
        rows.append(window_features(acc[start:stop], gyro[start:stop], fs))
        window_labels = np.asarray(activities[start:stop])
        values, counts = np.unique(window_labels, return_counts=True)
        labels.append(str(values[int(np.argmax(counts))]))
        centers.append((start + window / 2.0) / fs)
    return np.stack(rows), np.asarray(labels), np.asarray(centers)


def log_compress(features):
    """log10 of the LOG_FEATURES columns. The other columns pass through."""
    out = np.array(features, dtype=float, copy=True)
    columns = [FEATURE_NAMES.index(name) for name in LOG_FEATURES]
    out[:, columns] = np.log10(out[:, columns] + LOG_FLOOR)
    return out


class FeatureScaler:
    """Log of the magnitude features, then a training-set z-score.

    Test windows use the same mean and scale values as training. `log=False`
    keeps the plain z-score, which the notebook uses to show why the log is there.
    """

    def __init__(self, log=True):
        self.log = bool(log)
        self.mean_ = None
        self.scale_ = None

    def _prepare(self, features):
        return log_compress(features) if self.log else np.asarray(features, dtype=float)

    def fit(self, features):
        prepared = self._prepare(features)
        self.mean_ = prepared.mean(axis=0)
        scale = prepared.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, features):
        return (self._prepare(features) - self.mean_) / self.scale_
