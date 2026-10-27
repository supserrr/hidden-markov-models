# Activity recognition with a hidden Markov model

A Gaussian hidden Markov model, written from scratch in NumPy, that reads phone accelerometer and gyroscope data and decodes four activities: standing, walking, jumping, and still.

Formative 2 · Machine Learning Techniques · African Leadership University

<p align="center">
  <img src="results/figures/decoded_sequence.png" alt="Viterbi path following seven held-out recordings joined end to end" width="760">
</p>

## Results

Trained on 160 recordings and tested on 40 unseen ones, 581 one-second windows in all.

| Activity | Test windows | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Standing | 138 | 1.000 | 1.000 | 1.000 |
| Walking | 134 | 1.000 | 1.000 | 1.000 |
| Jumping | 149 | 1.000 | 1.000 | 1.000 |
| Still | 160 | 1.000 | 1.000 | 1.000 |

The test files come from the same person, phone, and session as the training files. So the result shows that the four activities are separable here, not that the model transfers to another person. The [report](report/Formative_2_Hidden_Markov_Models.docx) covers that, and what a transition matrix can learn from recordings that hold one activity each.

<p align="center">
  <img src="results/figures/transition.png" alt="Learned transition matrix" width="380">
  <img src="results/figures/confusion.png" alt="Confusion matrix on the unseen windows" width="380">
</p>

## Repository layout

```text
.
├── data/
│   ├── raw/<activity>/              Sensor Logger exports, one zip per recording, as recorded
│   ├── processed/<activity>/        the dataset: one labelled CSV per recording
│   ├── sequences/                   recordings that change activity (none yet)
│   ├── manifest.csv                 every CSV with its split, rate, duration, and source zip
│   └── README.md                    columns, units, and naming
├── notebooks/
│   └── hmm_activity_recognition.ipynb
├── src/
│   ├── convert.py                   zips to CSVs, plus the manifest
│   ├── features.py                  resampling, windows, nine features, log and z-score scaling
│   ├── hmm.py                       Gaussian HMM: Baum-Welch and Viterbi
│   ├── experiment.py                training, evaluation, and figures
│   └── data.py, evaluate.py, plots.py
├── results/
│   ├── figures/
│   └── metrics.json
└── report/
    └── Formative_2_Hidden_Markov_Models.docx
```

## Data

200 recordings made with Sensor Logger on one iPhone 16 Pro at 100 Hz: 50 per activity, 5.6 to 9.9 seconds each, and 393 to 439 seconds per activity. The last 10 recordings of each activity, by start time, are the test set. [data/README.md](data/README.md) describes the files, columns, and units.

## Videos

One face-visible video of the data collection per activity.

| Activity | Video |
| --- | --- |
| Standing | PASTE_LINK |
| Walking | PASTE_LINK |
| Jumping | PASTE_LINK |
| Still | PASTE_LINK |

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.convert
python -m src.experiment
```

`convert` turns the zips in `data/raw/` into the CSVs in `data/processed/` and rewrites `data/manifest.csv`, keeping any `mixed` rows already there. `experiment` fits the model on the training rows, scores the test rows, and writes `results/`. The notebook runs the same pipeline step by step, with the figures inline.

Mixed sequences are optional. Without a held-out one, the decoded-path figure joins held-out single-activity files end to end and says so in its title.

## Model

Hidden states are the four activities. Each observation is one 1.0 second window with a 0.5 second hop, after every file has been interpolated to 50 Hz. Gravity is the mean acceleration inside the window and is removed. The six magnitude and energy features go through log10, because they span about seven orders of magnitude between a phone on a desk and a jump. Then every feature is z-scored with the training windows only.

Emissions are diagonal Gaussians. Parameters start from the labeled training windows, then Baum-Welch updates them until the relative change in log-likelihood is below 10⁻⁶. Viterbi produces the label sequence that the sensitivity, specificity, and accuracy are counted on.
