# Hidden Markov model of four activity states

Formative 2 for Machine Learning Techniques. A diagonal Gaussian hidden Markov model, written in NumPy, reads windowed accelerometer and gyroscope features and decodes four activities: standing, walking, jumping, and still.

The CSV files in `data/` are **simulated** IMU streams (seed 7). They use the column layout the loader expects, at 40 Hz, 50 Hz, and 100 Hz, so resampling and training can be re-run without a phone. They are not recordings of a person. The brief also asks for four face-visible videos, one per activity. Those videos are not in this repository.

## Result

On 40 held-out single-activity files (585 windows) the Viterbi path is right on 99.7% of windows, and every file is right by majority vote. The only errors are two standing windows read as still. Baum-Welch raises the training log-likelihood from 11908.4 to 14768.9 in 9 iterations and does not swap the state names. Four held-out mixed recordings score 95.4% of windows.

## Layout

| Path | What it is |
| --- | --- |
| `data/raw/` | 50 CSV files per activity, 6.5–9.5 seconds each |
| `data/sequences/` | 16 longer files that chain activities |
| `data/manifest.csv` | Path, split, sampling rate, and `source=synthetic` |
| `src/hmm.py` | Baum-Welch and Viterbi |
| `src/features.py` | Resampling, windows, and the nine features |
| `notebooks/hmm_activity_recognition.ipynb` | The run, with outputs |
| `report/Formative_2_Hidden_Markov_Models.docx` | The short report |
| `figures/` | Magnitude, likelihood, transitions, confusion, decoded path |

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
python -m src.generate_data
python -m src.experiment
```

`generate_data` rewrites the CSVs from seed 7. `experiment` refits the model and refreshes `figures/` and `report/metrics.json`. Open the notebook to see the same steps with the plots inline.

## Model

Hidden states are the four activities. Each observation is one 1.0 second window with a 0.5 second hop, after every file has been interpolated to 50 Hz. Gravity is the mean acceleration inside the window and is removed. Features are z-scored with the training windows only.

Emissions are diagonal Gaussians. Parameters start from the labeled training windows, then Baum-Welch updates them until the relative change in log-likelihood is below 10⁻⁶. Viterbi produces the label sequence that the sensitivity, specificity, and accuracy are counted on.
