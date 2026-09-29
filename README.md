# ASL Alphabet Recognition with MobileNetV2

An experimental image classifier for the **26 letters of the American Sign Language (ASL) fingerspelling alphabet**. It uses a pretrained MobileNetV2 backbone for classification and includes a webcam demonstration with MediaPipe hand detection and spoken letter output.

> **Project status:** Research/prototype code. The repository contains a saved model and training log, but the dataset is not included, an independent test result has not been reported, and the webcam demonstration needs the fixes listed below. This project does not translate full ASL conversations or sentences.

## What is implemented

1. `data_loder.py` loads images from class folders, applies augmentation to training images, creates a validation split, and computes class weights.
2. `model_builder.py` builds a MobileNetV2 classifier with a frozen ImageNet-pretrained backbone, global average pooling, dropout, and a 26-class softmax output.
3. `train_model.py` trains the classifier and saves a best checkpoint, final model, CSV log, and accuracy/loss plots.
4. `asl_test.py` detects a hand in a webcam region using MediaPipe, predicts a letter from the image, and uses text-to-speech to announce it.

Training and webcam inference both use MobileNetV2's `preprocess_input` and 224 Ã— 224 RGB inputs.

## Results available now

The included `20250815_184817/training_log.csv` records a **best validation accuracy of 67.21% at epoch 26** (out of 30 logged epochs). The final logged validation accuracy is approximately **67.00%**. These are validation measurements, **not independent test-set results**. Training accuracy reached 100% in one epoch, but that number should not be used as a measure of performance on new signers or live webcam images.

Although `data_loder.py` constructs a test generator, the current training script reports a classification report for the **validation** set only. No held-out test accuracy, confusion matrix, signer-independent evaluation, or live webcam benchmark is provided yet.

## Setup and current run requirements

The code uses Python with TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, OpenCV, MediaPipe, and `pyttsx3`. Dependency versions have not yet been pinned or tested together across operating systems; a `requirements.txt` and verified setup instructions are planned.

For training, the code expects a dataset that is **not included** in this repository:

```text
ASL-Project-No-2/
â””â”€â”€ ASL_Alphabet_Dataset/
    â”œâ”€â”€ asl_alphabet_train/
    â”‚   â”œâ”€â”€ A/ ... images ...
    â”‚   â”œâ”€â”€ B/ ... images ...
    â”‚   â””â”€â”€ ... Z/
    â””â”€â”€ asl_alphabet_test/
        â”œâ”€â”€ A/ ... images ...
        â”œâ”€â”€ B/ ... images ...
        â””â”€â”€ ... Z/
```

The scripts use paths relative to the current working directory. Run them from inside `ASL-Project-No-2/`, after providing the data and installing compatible dependencies:

```bash
cd ASL-Project-No-2
python train_model.py
```

**Before training:** work on a copy of the dataset. Importing `data_loder.py` currently deletes exact duplicate files within each train/test directory tree and removes any `del`, `space`, and `nothing` class directories. It also creates plots and writes `class_indices.json` in the current working directory.

**Before running the webcam demo:** fix the model path in `asl_test.py`. It currently loads `saved_models/20250815_184817/best_model.h5`, while the committed checkpoint is at `../20250815_184817/best_model.h5` when running from `ASL-Project-No-2/`. The demo also draws landmarks on the image *before* sending it to the classifier and speaks a test phrase during every accepted prediction; these behaviors should be corrected before relying on it.

## What we need to do next

### 1. Make the demo run reliably

- [ ] Load the saved model and `class_indices.json` using paths based on the script location or command-line arguments.
- [ ] Keep the image sent to MobileNetV2 free of MediaPipe drawings; draw landmarks only on the displayed frame.
- [ ] Remove the startup/test speech and the phrase spoken for every prediction. Speak a stable predicted letter only after an appropriate cooldown.
- [ ] Replace the fixed webcam region with checked bounds or a hand-centered crop, and handle missing camera frames gracefully.
- [ ] Add a meaningful rejection rule for uncertain predictions and measure prediction latency/FPS.

### 2. Establish trustworthy evaluation

- [ ] Document the dataset source, license, number of samples per class, and how train/validation/test sets were made.
- [ ] Check for identical or near-duplicate images **across** splits; where signer identities are available, evaluate on signers absent from training.
- [ ] Evaluate the saved best model on an untouched test set. Report overall accuracy, per-class precision/recall/F1, and a confusion matrix.
- [ ] Test with new webcam users, lighting, backgrounds, and hand positions. Report these results separately from dataset test accuracy.
- [ ] Inspect difficult letter pairs and record examples of failure cases.

### 3. Improve the training pipeline

- [ ] Move data cleaning into an explicit, non-destructive preparation script instead of deleting files on import.
- [ ] Verify that train, validation, and test sets use the same Aâ€“Z label mapping and preprocessing.
- [ ] Review augmentations, particularly horizontal flips, to ensure they preserve the intended sign label.
- [ ] Add a reproducible environment, seed handling, configuration, and training instructions.
- [ ] Compare the frozen-backbone baseline with careful partial fine-tuning after the evaluation protocol is fixed.

### 4. Clarify the recognition scope

- [ ] Treat this as **fingerspelling letter classification**, rather than general ASL translation.
- [ ] Explore short video sequences for motion-dependent letters such as **J** and **Z**; a single image cannot represent their complete movement.
- [ ] If expanding to words or sentences, define a new dataset and evaluation protocol for that task.

## License

See [LICENSE](LICENSE) for the MIT license.
