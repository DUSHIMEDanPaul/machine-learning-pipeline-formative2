# machine-learning-pipeline-formative2

**Member A — Data & Recommendation Lead**
Owns the tabular side, which is the heaviest single track:

Merge customer_social_profiles + customer_transactions (decide the join key, handle unmatched rows).
Feature engineering on the merged set — capture the relevant signals from both sources, not just one.
Build and evaluate the Product Recommendation Model (RF / LogReg / XGBoost) predicting the product a customer would buy. Report Accuracy, F1, loss.
Produces the merged dataset deliverable for the repo.

**Member B — Vision / Facial Recognition Lead**
Owns the image pipeline end to end:

Loading + displaying sample images for all members.
Augmentation code (rotation, flip, grayscale) applied per image.
Feature/embedding extraction → writes image_features.csv in the agreed schema.
Trains and evaluates the Facial Recognition Model (face matches a known user vs. not).

**Member C — Audio / Voiceprint Lead**
Mirror of B, on the audio side:

Loading + displaying waveforms and spectrograms for each member's samples.
At least two augmentations per sample (pitch shift, time stretch, background noise).
Feature extraction (MFCCs, spectral roll-off, energy) → writes audio_features.csv.
Trains and evaluates the Voiceprint Verification Model (approved voice vs. not).

**Member D — Integration & Systems Lead**
Lighter early, heavy at the end — so front-load repo work while A/B/C are still collecting data:

Scaffold the GitHub repo structure and a shared data-loading utility everyone imports.

**image_features.csv**
columnexamplenotesmember_iddanshort, lowercase, consistent everywhereimage_iddan_smiling_01unique per rowexpressionneutral / smiling / surprisedaugmentationoriginal / rotate / flip / grayscaleoriginal = the untouched imagelabeldantarget for the face model (who this is)feat_0 … feat_n0.0123, …your embedding/histogram vector, fixed length
So the header is: member_id, image_id, expression, augmentation, label, feat_0, feat_1, …, feat_n



**audio_features.csv**
Same shape. For MFCCs, take the mean across time frames so you get a fixed 13 columns instead of a variable-length matrix — this is the "easy" version and totally standard.
columnexamplenotesmember_iddanclip_iddan_approve_01phraseyes_approve / confirm_transactionaugmentationoriginal / pitch_shift / time_stretch / noiselabeldantarget for the voiceprint modelmfcc_0 … mfcc_12-245.3, …mean of each MFCC over timespectral_rolloff3421.5single mean valueenergy0.018RMS energy, single value
Header: member_id, clip_id, phrase, augmentation, label, mfcc_0, …, mfcc_12, spectral_rolloff, energy
Build the command-line app that orchestrates the full flow: face image → gates the model call → voice → approves → runs prediction.
Implement the two "access denied" pathways (unauthorized face, unauthorized voice) and the unauthorized-attempt simulations.
Own the report skeleton and the demo video recording, chasing everyone for their contribution section.
