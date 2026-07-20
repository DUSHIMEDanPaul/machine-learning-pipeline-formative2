# User Identity and Product Recommendation System

Multimodal authentication pipeline that verifies a user's **face**, then their **voice**, before granting access to a **product recommendation** model. Built as the Formative 2 group assignment for Machine Learning at African Leadership University.

**Report:** `ML_Group_Assignment_Report.pdf` · **Demo video:** [Google Drive folder](https://drive.google.com/drive/folders/1sjDRELqmJtCrTvMBYFe3NmJvv4Z92QNe?usp=sharing)

## Team

| Member | Task |
|---|---|
| Enock MUGISHA | Task 1 — Data merge, EDA, and feature engineering |
| Ulrich RUKAZAMBUGA | Task 2 — Image data collection and processing |
| Hugues MUNEZERO | Task 3 — Sound data collection and processing |
| Dan Paul DUSHIME | Task 4 — Model creation and evaluation; repo scaffold and README |

All members contributed their own facial images (neutral, smiling, surprised) and voice recordings ("Yes, approve" / "Confirm transaction").

## How the system works

```
face image ──> [Gate 1: Facial Recognition] ──DENY──> exit
                        │ verified as member M
                        v
voice clip ──> [Gate 2: Voiceprint Verification] ──DENY──> exit
                        │ approved AND speaker == M      (low confidence, or
                        v                                 face/voice mismatch)
               [Product Recommendation] ──> top-3 predicted products
```

Gate 2 rejects two distinct failure modes: an unapproved voice, **and** a legitimate voice that doesn't match the identity established at Gate 1 (e.g. Dan's face presented with Enock's voice).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── customer_social_profiles.xlsx      # provided source data
│   ├── customer_transactions.xlsx         # provided source data
│   ├── merged_dataset.csv                 # Task 1 output (117 rows x 32 cols)
│   ├── image_features.csv                 # Task 2 output (48 rows x 1285 cols)
│   └── audio_features.csv                 # Task 3 output (32 rows x 20 cols)
├── notebooks/
│   ├── task1_data_merge_eda.ipynb
│   ├── task2_image_processing.ipynb
│   ├── task3_audio_processing.ipynb
│   └── task4_model_training.ipynb
├── scripts/
│   ├── merged_dataset_script.py           # Task 1 pipeline
│   ├── image_python_scripts.py            # Task 2 pipeline
│   ├── audio_python_scripts.py            # Task 3 pipeline
│   └── train_models.py                    # Task 4 training (all three models)
├── models/
│   ├── product_model.pkl
│   ├── face_model.pkl
│   └── voice_model.pkl
├── unauthorized/
│   ├── unknown_picture.png                # stranger face for denial demo
│   └── unknown_voice.wav                  # stranger voice for denial demo
└── auth_app.py                            # command-line demonstration app
```

## Environment and installation

Developed in Google Colab (Python 3.12). To run locally:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
scikit-learn==1.6.1
tensorflow
librosa
soundfile
Pillow
matplotlib
openpyxl
```

> **The scikit-learn pin matters.** The `.pkl` model artifacts were serialized under scikit-learn 1.6.1. Loading them under a different version may emit `InconsistentVersionWarning` or fail outright. Install the pinned version before running `auth_app.py`.

`ffmpeg` is required for the audio pipeline (pre-installed in Colab; `apt install ffmpeg` locally).

## Reproducing the pipeline

Run in this order. Each stage's output feeds the next.

**1. Data merge and EDA (Task 1)** — `notebooks/task1_data_merge_eda.ipynb`
Profiles both Excel sources (summary statistics, types, missing values), produces three labeled EDA figures (distributions, outlier boxplots, correlations + class balance), normalizes the mismatched customer IDs (`A178` → `178`), aggregates the social data to one row per customer to prevent a cartesian-product merge, inner-joins on customer ID, engineers date / behavioural / cross-source interaction features, and writes `merged_dataset.csv`.

**2. Image features (Task 2)** — `notebooks/task2_image_processing.ipynb`
Loads 12 member photos (4 members × 3 expressions), applies 3 augmentations per image (rotation, brightness, grayscale), extracts 1280-dimensional MobileNetV2 embeddings (ImageNet weights, frozen, global average pooling, 224×224 input), and writes `image_features.csv`.

**3. Audio features (Task 3)** — `notebooks/task3_audio_processing.ipynb`
Converts mixed-format recordings (OGG / MPEG / MP4) to mono 22 050 Hz WAV with ffmpeg, plots waveform + spectrogram per clip, applies 3 augmentations (pitch shift ±2 semitones, 1.15× time stretch, Gaussian noise), extracts 13 MFCC means + spectral roll-off + RMS energy, and writes `audio_features.csv`.

**4. Model training (Task 4)** — `notebooks/task4_model_training.ipynb`
Trains all three models and saves the pickle artifacts described below.

## Models and evaluation

All models: **Random Forest (300 trees, seed 42)** on standardized features. A Logistic Regression baseline was also trained for the product task.

Split design avoids augmentation leakage — augmented copies of one source photo/clip never straddle the train/test boundary:

| Model | Features | Test protocol | n | Accuracy | F1 (macro) | Log loss |
|---|---|---|---|---|---|---|
| Facial recognition | MobileNetV2 embeddings | hold out "surprised" expression | 16 | **1.000** | 1.000 | 0.783 |
| Voiceprint verification | MFCC + roll-off + energy | train "yes approve", test "confirm transaction" | 16 | **0.875** | 0.874 | 0.737 |
| Product recommendation | merged dataset | 75/25 stratified | 30 | **0.300** | 0.296 | 1.845 |

The product model's near-chance result is consistent with the EDA: predictors in the synthetic source data are essentially uncorrelated with the target (r ≈ 0.03–0.09). Full analysis, including the histogram-vs-embedding feature study and the open-set rejection findings, is in the report (Sections 5–6).

### Pickle artifact format

Every `.pkl` is a dict, loadable through one code path:

```python
{
  "model":         fitted RandomForestClassifier,
  "scaler":        fitted StandardScaler,
  "feature_cols":  ordered list of training column names,
  "label_encoder": LabelEncoder,   # product model only
}
```

## Running the demonstration (`auth_app.py`)

```bash
# 1. Authorized transaction — both gates pass, recommendation shown
python auth_app.py --face dan_neutral.jpeg --voice dan_yes_approve.wav

# 2. Unauthorized face — denied at Gate 1
python auth_app.py --face unauthorized/unknown_picture.png --voice dan_yes_approve.wav

# 3. Unauthorized voice — face passes, denied at Gate 2
python auth_app.py --face dan_neutral.jpeg --voice unauthorized/unknown_voice.wav

# 4. Identity mismatch — legitimate face + different member's voice, denied at Gate 2
python auth_app.py --face dan_neutral.jpeg --voice enock_yes_approve.wav
```

These four commands are the exact scenarios shown in the demo video.

**Thresholds** (set in `auth_app.py`): `FACE_THRESHOLD = 0.60`, `VOICE_THRESHOLD = 0.50`, plus a top-1 vs top-2 margin check. The face threshold was calibrated empirically: enrolled members' neutral/smiling images score ≈ 0.9 while the unauthorized image scores 0.493. Note that the face and voice models are closed-set classifiers — they always answer with one of the four members — so the confidence threshold is what makes rejection of strangers possible at all. The report's Section 6 documents the limits of this approach.

## Known limitations

- **Tiny dataset**: 12 photos and 8 recordings total. All metrics carry high variance; the perfect face score demonstrates the approach, not a production error rate.
- **Open-set rejection is weak**: generic ImageNet embeddings encode scene content, not facial identity, so stranger rejection relies on threshold calibration rather than representation quality. The principled fix is a face-specific embedding network (ArcFace/FaceNet) and, on the audio side, speaker embeddings (x-vectors).
- **Mild leakage in the product features**: per-customer spend aggregates are computed over the full dataset before splitting. A stricter pipeline would compute them within training folds only.
- **Pickle portability**: artifacts are version-sensitive; use the pinned scikit-learn.

## Data and privacy note

The face images and voice recordings in this repository belong to the four team members and are included solely for grading of this assignment. The unauthorized samples were provided with the subject's consent. Please do not reuse any biometric data from this repository for other purposes.
