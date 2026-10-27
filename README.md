# Hidden Markov model of four activity states

Formative 2 for Machine Learning Techniques. A diagonal Gaussian hidden Markov model, written in NumPy, reads windowed accelerometer and gyroscope features and decodes four activities: standing, walking, jumping, and still.

The data is 200 Sensor Logger recordings from one iPhone 16 Pro at 100 Hz: 50 per activity, each 5.6 to 9.9 seconds long. The last 10 recordings of each activity are held out for testing.

## Layout

| Path | What it is |
| --- | --- |
| `RECORDINGS/<Activity>/` | Sensor Logger zips, one per recording, as exported from the phone |
| `data/raw/<activity>/` | One CSV per recording, written by `python -m src.convert`. `activity` is standing, walking, jumping, or still |
| `data/sequences/` | Optional longer files that chain activities. Set `activity` to `mixed` in the manifest |
| `data/manifest.csv` | Path, activity, split (`train` or `test`), sample rate, duration, and the zip each CSV came from |
| `src/convert.py` | Sensor Logger zips to project CSVs, plus the manifest |
| `src/hmm.py` | Baum-Welch and Viterbi |
| `src/features.py` | Resampling, windows, the nine features, and the log and z-score scaling |
| `notebooks/hmm_activity_recognition.ipynb` | The pipeline |
| `report/Formative_2_Hidden_Markov_Models.docx` | The short report |
| `figures/` | Written by `python -m src.experiment` |

Every CSV under `data/` uses this header:

```text
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,activity
```

`timestamp` is seconds. Acceleration is m/s² with gravity removed, as Sensor Logger reports it on iOS. Gyroscope is rad/s. `activity` is `standing`, `walking`, `jumping`, or `still`, and may change from row to row inside a mixed sequence.

## Videos

Paste one face-visible link per activity. A missing video is an immediate zero on the brief.

| Activity | Link |
| --- | --- |
| Standing | PASTE_LINK |
| Walking | PASTE_LINK |
| Jumping | PASTE_LINK |
| Still | PASTE_LINK |

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.convert
python -m src.experiment
```

`convert` rewrites `data/raw/` and `data/manifest.csv` from the zips in `RECORDINGS/`, and keeps any `mixed` rows already in the manifest. `experiment` reads the manifest, refits the model, and writes `figures/` plus `report/metrics.json`. The manifest columns are `path`, `activity`, `split`, `source`, `sample_rate_hz`, `duration_s`, `n_rows`, and `origin`. `path` is relative to the repository root. `split` is `train` or `test`.

Mixed sequences are optional. Without a held-out one, the decoded-path figure joins held-out single-activity files end to end and says so in its title.

## Model

Hidden states are the four activities. Each observation is one 1.0 second window with a 0.5 second hop, after every file has been interpolated to 50 Hz. Gravity is the mean acceleration inside the window and is removed. The six magnitude and energy features go through log10, because they span about seven orders of magnitude between a phone on a desk and a jump. Then every feature is z-scored with the training windows only.

Emissions are diagonal Gaussians. Parameters start from the labeled training windows, then Baum-Welch updates them until the relative change in log-likelihood is below 10⁻⁶. Viterbi produces the label sequence that the sensitivity, specificity, and accuracy are counted on.
