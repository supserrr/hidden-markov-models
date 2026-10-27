# Dataset

200 recordings of four activities, made with Sensor Logger 1.68 on one iPhone 16 Pro on 27 October 2026. The accelerometer and the gyroscope were both sampled every 10 ms (100 Hz).

| Activity | How it was recorded | Files | Total | Length of one file |
| --- | --- | ---: | ---: | --- |
| Standing | phone held steady at waist level, screen up | 50 | 438.5 s | 5.58–9.87 s |
| Walking | steady pace, phone held the same way | 50 | 393.0 s | 7.21–8.68 s |
| Jumping | continuous jumps | 50 | 399.0 s | 6.71–9.46 s |
| Still | phone lying on a flat surface | 50 | 438.0 s | 7.65–9.93 s |

## Folders

- `raw/<activity>/<start time>.zip` is each export exactly as Sensor Logger wrote it. The name is the start time in UTC. A zip holds `Accelerometer.csv`, `Gyroscope.csv`, their uncalibrated versions, `Metadata.csv`, and an empty `Annotation.csv`.
- `processed/<activity>/<activity>_NNN.csv` is the same recording after `python -m src.convert`. `NNN` counts the recordings of that activity in start-time order, so `standing_001.csv` is the first standing recording.
- `sequences/` is for recordings that change activity mid-file, with a label on every row. There are none yet.
- `unseen/` is a later session on the same phone, from 22:29 to 22:36 UTC on 27 October, used only for testing. `unseen/raw/<activity>/` holds 5 single-activity zips per activity and `unseen/raw/mixed/` holds 2 continuous recordings of about 61 s that change activity. `python -m src.convert` writes them to `unseen/processed/` with split `unseen`.
- `unseen/segments.csv` labels the continuous recordings. Each row is `recording,start_s,end_s,activity`, in seconds from the start of that zip, and the spans of one recording must meet end to end. `transition` marks handling between activities, such as putting the phone down: those windows are decoded but not scored.
- `manifest.csv` has one row per CSV and is what training reads.

## CSV columns

| Column | Unit | Meaning |
| --- | --- | --- |
| `timestamp` | s | Time since the first sample |
| `acc_x`, `acc_y`, `acc_z` | m/s² | User acceleration: iOS has already removed gravity |
| `gyro_x`, `gyro_y`, `gyro_z` | rad/s | Rotation rate, put on the accelerometer timestamps |
| `activity` | | `standing`, `walking`, `jumping`, or `still`, plus `transition` in a continuous recording |

## Manifest columns

| Column | Meaning |
| --- | --- |
| `path` | CSV path from the repository root |
| `activity` | The activity, or `mixed` for a file in `sequences/` or `unseen/processed/mixed/` |
| `split` | `train`, `test`, or `unseen`. The last 10 recordings of each activity are `test`; everything in `unseen/` is `unseen` |
| `source` | `sensor-logger` |
| `sample_rate_hz` | Median sample rate of the file |
| `duration_s` | Length of the file |
| `n_rows` | Number of samples |
| `origin` | The zip in `raw/` that the CSV came from |
