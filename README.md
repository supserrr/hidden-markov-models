# Hidden Markov model of four activity states

Formative 2 for Machine Learning Techniques. A diagonal Gaussian hidden Markov model, written in NumPy, reads windowed accelerometer and gyroscope features and decodes four activities: standing, walking, jumping, and still.

There are no recordings in this repository. Put real sensor exports in `data/raw/<activity>/` and list them in `data/manifest.csv`. The brief also asks for four face-visible videos, one per activity. Those videos are not in this repository.

## Layout

| Path | What it is |
| --- | --- |
| `data/raw/<activity>/` | One CSV per recording. `activity` is standing, walking, jumping, or still |
| `data/sequences/` | Optional longer files that chain activities. Set `activity` to `mixed` in the manifest |
| `data/manifest.csv` | Path, activity, split (`train` or `test`), sample rate, and duration |
| `src/hmm.py` | Baum-Welch and Viterbi |
| `src/features.py` | Resampling, windows, and the nine features |
| `notebooks/hmm_activity_recognition.ipynb` | The pipeline |
| `report/Formative_2_Hidden_Markov_Models.docx` | The short report |
| `figures/` | Written by `python -m src.experiment` after recordings are in place |

A replacement recording must use this header:

```text
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,activity
```

`timestamp` is seconds. Acceleration is m/s². Gyroscope is rad/s. `activity` is `standing`, `walking`, `jumping`, or `still`.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.experiment
```

`experiment` reads `data/manifest.csv`, refits the model, and writes `figures/` plus `report/metrics.json`. The manifest columns are `path`, `activity`, `split`, `source`, `sample_rate_hz`, `duration_s`, and `n_rows`. `path` is relative to the repository root. `split` is `train` or `test`.

## Model

Hidden states are the four activities. Each observation is one 1.0 second window with a 0.5 second hop, after every file has been interpolated to 50 Hz. Gravity is the mean acceleration inside the window and is removed. Features are z-scored with the training windows only.

Emissions are diagonal Gaussians. Parameters start from the labeled training windows, then Baum-Welch updates them until the relative change in log-likelihood is below 10⁻⁶. Viterbi produces the label sequence that the sensitivity, specificity, and accuracy are counted on.
