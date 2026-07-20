"""
User Identity & Product Recommendation System
=============================================
Flow: face -> voice -> product prediction, with denial at each gate.

Usage:
    python auth_app.py --face dan_neutral.jpeg --voice dan_yes_approve.wav

Requires (same directory unless paths are given):
    face_model.pkl, voice_model.pkl, product_model.pkl, merged_dataset.csv
"""
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import librosa
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------- thresholds
FACE_THRESHOLD = 0.60    # unknown face scores 0.493; members' neutral/smiling ~0.9
VOICE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.20  # gap between top-1 and top-2 probability
SR = 22050

# ------------------------------------------------------- feature extractors
# NOTE: these must stay byte-identical to the extractors used in the
# training notebooks, or the models will receive mismatched inputs.
_embedder = MobileNetV2(weights="imagenet", include_top=False,
                        pooling="avg", input_shape=(224, 224, 3))


def image_features(path):
    """1280-d MobileNetV2 embedding, matching the Task 2 pipeline."""
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = preprocess_input(np.array(img, dtype=np.float32))
    return _embedder.predict(np.expand_dims(arr, 0), verbose=0)


def audio_features(path):
    """13 MFCC means + spectral roll-off + RMS energy, matching Task 3."""
    y, sr = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    energy = float(np.sqrt(np.mean(y ** 2)))
    return np.concatenate([mfcc, [rolloff, energy]]).reshape(1, -1)


# --------------------------------------------------------------- utilities
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def identify(art, X):
    """Return (best_label, confidence, top1-top2 margin) for one sample."""
    Xs = art["scaler"].transform(X) if art.get("scaler") is not None else X
    if art.get("pca") is not None:
        Xs = art["pca"].transform(Xs)
    proba = art["model"].predict_proba(Xs)[0]
    order = np.argsort(proba)[::-1]
    return (art["model"].classes_[order[0]],
            float(proba[order[0]]),
            float(proba[order[0]] - proba[order[1]]))


def banner(text):
    print("\n" + "=" * 55)
    print(text)
    print("=" * 55)


# ------------------------------------------------------------------- gates
def gate_face(path):
    banner("STEP 1 - FACIAL RECOGNITION")
    who, conf, margin = identify(load("face_model.pkl"), image_features(path))
    print(f"Input image: {path}")
    print(f"Best match : {who}   conf={conf:.3f}  margin={margin:.3f}"
          f"   (thresholds: conf {FACE_THRESHOLD}, margin {MARGIN_THRESHOLD})")

    if conf < FACE_THRESHOLD or margin < MARGIN_THRESHOLD:
        print("\n  >>> ACCESS DENIED - face not recognised as a known user.")
        return None
    print(f"\n  >>> FACE VERIFIED - welcome, {who}.")
    print("      You may now request a product prediction.")
    return who


def gate_voice(path, expected_user):
    banner("STEP 2 - VOICEPRINT VERIFICATION")
    who, conf, margin = identify(load("voice_model.pkl"), audio_features(path))
    print(f"Input audio: {path}")
    print(f"Best match : {who}   conf={conf:.3f}  margin={margin:.3f}"
          f"   (threshold: conf {VOICE_THRESHOLD})")

    if conf < VOICE_THRESHOLD:
        print("\n  >>> ACCESS DENIED - voice is not an approved sample.")
        return False
    if who != expected_user:
        print(f"\n  >>> ACCESS DENIED - voice belongs to '{who}', but the face "
              f"identified '{expected_user}'. Identity mismatch.")
        return False
    print(f"\n  >>> VOICE APPROVED - transaction confirmed by {who}.")
    return True


def run_prediction(user):
    banner("STEP 3 - PRODUCT RECOMMENDATION")
    art = load("product_model.pkl")
    df = pd.read_csv("merged_dataset.csv")
    row = df[art["feature_cols"]].astype(float).iloc[[0]]
    print("(using sample customer row 0 from merged_dataset.csv)")

    proba = art["model"].predict_proba(art["scaler"].transform(row.values))[0]
    le = art["label_encoder"]
    order = np.argsort(proba)[::-1]

    print(f"\nRecommendations for {user}:")
    for rank, i in enumerate(order[:3], 1):
        print(f"   {rank}. {le.classes_[i]:<14} {proba[i] * 100:5.1f}%")
    print(f"\n  >>> PREDICTED PRODUCT: {le.classes_[order[0]]}")


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Identity-gated product recommendation system")
    parser.add_argument("--face", required=True, help="path to face image")
    parser.add_argument("--voice", required=True, help="path to voice clip")
    args = parser.parse_args()

    banner("USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM")

    user = gate_face(args.face)
    if user is None:
        print("\nTransaction terminated at face gate.\n")
        sys.exit(1)

    if not gate_voice(args.voice, user):
        print("\nTransaction terminated at voice gate.\n")
        sys.exit(1)

    run_prediction(user)
    print("\nTransaction complete.\n")


if __name__ == "__main__":
    main()
