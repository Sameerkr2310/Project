# Handwritten Digit Recognition

**CNN-based digit classification with an interactive Streamlit web application**

Recognises a handwritten digit (0–9) from an **uploaded image** or from a digit **drawn on a canvas**.
A convolutional neural network is trained on the MNIST dataset, and a preprocessing pipeline turns any
user image into an MNIST-style input. The app shows the predicted digit, the confidence, the runner-up
digit and the exact 28 × 28 image that the network received.

Everything runs on your own computer. No API key, cloud account or paid service is needed.

> Course project (UCS7702PJ), B.Tech Computer Science & Engineering, Shri Ramswaroop Memorial University, Lucknow.

---

## Why this project?

MNIST digits are white on black, scaled to a 20 × 20 box and centred by their centre of mass. Digits that
a user draws or photographs look different (dark ink on paper, any size, any position, any pen thickness),
so a model that scores well on MNIST can still fail on real handwriting. This project closes that gap with
a preprocessing pipeline, confidence feedback, and a separate test on **your own** handwriting.

## Features

- **Two input modes:** upload a PNG / JPG / JPEG, or draw on a canvas (adjustable brush size).
- **MNIST-style preprocessing:** polarity detection, noise removal, cropping, stroke-width normalisation,
  resizing to 20 px and centring by centre of mass on a 28 × 28 canvas.
- **Explained predictions:** predicted digit, confidence, runner-up digit, probability chart and the
  processed 28 × 28 input.
- **Warnings:** low confidence (below 60%) and ambiguous predictions (two close digits, e.g. 4 vs 9).
- **Evaluation dashboard:** accuracy, loss, per-digit precision / recall / F1, confusion matrix,
  training curves and the most confused digit pairs.
- **Own-handwriting test:** save a drawing with its correct digit and measure real-world accuracy with
  `evaluate_custom.py`.
- **Readable errors:** missing model, missing or broken TensorFlow, damaged model file, blank or tiny
  images are all reported with instructions instead of a traceback.
- **Reproducible and tested:** fixed seed, saved metrics, one configuration file, 23 automated tests.

## Quick start

Requires **Python 3.9 – 3.12** (TensorFlow 2.16 / 2.17 does not support Python 3.13 or newer).

| OS | Command |
|----|---------|
| Windows | double-click `run.bat` |
| Linux / macOS | `./run.sh` |

The launchers create / use a virtual environment, install the requirements, train the model only if
`model/digit_model.keras` is missing, and then start the app.

**Manual steps**

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
python train.py                  # trains the model and writes model/ and results/
python -m streamlit run app.py   # opens the web app in your browser
```

## Using the app

| Page | What you can do |
|------|-----------------|
| **Home** | Project description, model architecture and preprocessing steps |
| **Digit Prediction** | Upload an image of one digit and see the prediction, confidence, runner-up and processed input |
| **Draw Digit** | Draw a digit (brush size 8 – 28, **12 – 16 works best**), see the result, and save a wrongly predicted drawing with its correct digit |
| **Model Performance** | Accuracy, per-digit report, confusion matrix, curves, confused pairs and own-digit accuracy |

**Tips for good predictions:** one digit per image, drawn large so that it fills most of the canvas, plain
background. Very thick brushes merge strokes, and a closed loop in a 5 can look like a 0 — check the
processed 28 × 28 image to see what the network really received.

## How it works

```
Upload / canvas -> Validate -> Preprocess (28x28) -> CNN -> Softmax -> Digit + confidence + warnings
```

**Preprocessing (`utils/preprocessing.py`)**

1. Reject empty, tiny (< 8 × 8 px) or blank (almost no contrast) images.
2. Convert to grayscale and stretch the contrast.
3. Detect polarity from the border pixels and invert so the digit is white on black.
4. Remove faint noise and crop to the digit's bounding box.
5. Estimate the stroke width (distance transform) and thin or thicken it to about 2.6 px.
6. Resize so the longest side is 20 px and paste onto a 28 × 28 canvas.
7. Shift so the centre of mass is at the centre (as in MNIST), scale to 0 – 1.

**Prediction (`utils/predictor.py`)** returns the top-3 digits and the margin between the first and the
second. Confidence below 60% or a margin below 0.25 triggers a warning.

## Model

Improved model (**version 2**, `train.py`):

| Stage | Layers |
|-------|--------|
| Augmentation (training only) | random rotation 8%, shift 8%, zoom 10% |
| Block 1 | 2 × [Conv 32, 3×3, BatchNorm, ReLU] → MaxPool 2×2 → Dropout 0.25 |
| Block 2 | 2 × [Conv 64, 3×3, BatchNorm, ReLU] → MaxPool 2×2 → Dropout 0.25 |
| Classifier | Flatten → Dense 128 (ReLU) → Dropout 0.4 → Dense 10 (Softmax) |

About 470,000 parameters. Training: Adam (lr 0.001), sparse categorical cross-entropy, batch size 128,
up to 25 epochs, early stopping (patience 5, best weights restored), `ReduceLROnPlateau` (factor 0.5,
patience 2), 10% of the training images held out for validation, seed 42.

```bash
python train.py                    # defaults
python train.py --epochs 15        # quicker run
python train.py --seed 7           # different random seed
```

## Results

| Model | Test accuracy (10,000 MNIST images) | Test loss | Parameters | Status |
|-------|-------------------------------------|-----------|------------|--------|
| Baseline (version 1: 2 conv layers) | **98.84%** | 0.0349 | 225,034 | measured, 8 epochs (early stopping) |
| Improved (version 2) | target about 99% or higher | – | about 470,000 | run `python train.py` to measure |

> The `model/digit_model.keras` file and the numbers in `results/` that come with this repository are from the
> **baseline** model. After you run `python train.py`, the files are replaced by the improved model's results,
> and the Model Performance page stops showing the "version 1" notice.
> Accuracy comes from a single run with a fixed seed and may differ slightly between runs.

Files written by `train.py` in `results/`: `metrics.json`, `history.csv`, `classification_report.csv`,
`accuracy.png`, `loss.png`, `confusion_matrix.png`, `most_confused.csv`, `architecture.txt`.

## Test on your own handwriting

MNIST accuracy does not guarantee accuracy on your drawings. Save labelled images in `custom_digits/` in
either layout:

```
custom_digits/7/photo1.png        # folder name = true digit
custom_digits/7_002.png           # or a file name that starts with the digit
```

(The Draw page can save drawings here for you.) Then run:

```bash
python evaluate_custom.py
```

It prints the overall and per-digit accuracy and every mistake, and writes `results/custom_eval.json`,
which the Model Performance page shows.

## Project structure

```
.
├── app.py                  Streamlit app (Home, Digit Prediction, Draw Digit, Model Performance)
├── train.py                Builds, trains and evaluates the CNN; writes model/ and results/
├── evaluate_custom.py      Measures accuracy on the user's own digits
├── config.py               Paths, seed, epochs, batch size, confidence thresholds
├── requirements.txt        Python dependencies
├── run.bat / run.sh        One-command launchers (Windows / Linux, macOS)
├── utils/
│   ├── preprocessing.py    Image -> 28x28 MNIST-style input
│   ├── predictor.py        Model loading, prediction, top-3 and margin, readable errors
│   └── custom_samples.py   Saves a drawing with its correct digit
├── tests/                  Automated tests (unittest)
├── assets/sample_images/   Sample digit images for a quick demo
├── model/                  Saved network (digit_model.keras)
├── results/                Metrics, reports and charts
└── custom_digits/          Your own labelled digits
```

## Configuration

Settings live in `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `SEED` | 42 | Random seed for repeatable training |
| `DEFAULT_EPOCHS` | 25 | Maximum number of epochs |
| `DEFAULT_BATCH_SIZE` | 128 | Images per weight update |
| `LOW_CONFIDENCE` | 0.60 | Below this the app asks for a clearer drawing |
| `AMBIGUOUS_MARGIN` | 0.25 | Below this gap between the top two digits the app warns |
| `MODEL_VERSION` | 2 | Marks the improved architecture |

## Tests

```bash
python -m unittest discover tests -v
```

23 tests: preprocessing (polarity, centring, noise, image modes and sizes, blank / tiny images) and the
predictor (result fields, top-3 ranking, margin, model failures, missing or broken TensorFlow, saving of
drawings, loading of custom digits). The predictor is tested with a fake model, so almost all tests run
**without TensorFlow**; one optional smoke test checks the saved model on the sample images.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Trained model not found" | Run `python train.py` once, or use the train button in the app |
| "TensorFlow is installed but could not start" | Use Python 3.9 – 3.12 in a new virtual environment and reinstall the requirements |
| Windows: *"An Application Control policy has blocked this file"* (DLL load failed, `_pywrap_analyzer_wrapper`) | A Windows security setting (Smart App Control or a college / company policy) blocks a TensorFlow library. Use Python 3.12 in a new virtual environment. On your own PC you can turn Smart App Control off (Windows Security → App & browser control; this cannot be undone without resetting Windows). On a managed PC use another computer, WSL2 or Google Colab |
| A drawn digit is predicted wrongly | Draw one large digit with brush size 12 – 16 and check the processed 28 × 28 image |
| Low-confidence warning | Draw a clearer, larger single digit; several digits in one image are not supported |

## Limitations

- One digit per image; letters and symbols are not recognised.
- Trained on MNIST, so unusual writing styles, faint, blurred or cluttered images may be misread.
- Similar pairs such as 4/9 and 3/5 can still be confused.
- Reported accuracy is from a single run with a fixed seed.
- TensorFlow 2.16 / 2.17 needs Python 3.9 – 3.12.

## Future work

- Multi-digit recognition (segment connected components, predict each part).
- Fine-tune on handwriting collected through the Draw page.
- Letters and symbols with the EMNIST dataset.
- Export to TensorFlow Lite or ONNX for mobile and browser use; Docker deployment.

## Built with

Python · TensorFlow / Keras · NumPy · SciPy · Pillow · scikit-learn · pandas · Matplotlib · Streamlit ·
streamlit-drawable-canvas · unittest

## Team

**Tanishq Singh** (202310101150003) · **Sameer Sagar** (202310101150007) · **Harsh Trigunayat** (202310101150008)

Project guide: **Er. Arun Maurya**
Faculty of Computer Science & Engineering, Institute of Technology, Shri Ramswaroop Memorial University, Lucknow.

## References

1. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-Based Learning Applied to Document Recognition",
   Proceedings of the IEEE, 1998 (LeNet-5 and MNIST).
2. S. Ioffe and C. Szegedy, "Batch Normalization", ICML, 2015.
3. N. Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR, 2014.
4. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR, 2015.
5. P. Y. Simard, D. Steinkraus and J. C. Platt, "Best Practices for Convolutional Neural Networks Applied to
   Visual Document Analysis", ICDAR, 2003.
