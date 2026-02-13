# Handwriting Compare v1

Compare two handwriting generation approaches on a shared EMNIST letters domain:
- Offline image generation: **DCGAN** (`28x28` grayscale letters)
- Online-like sequence generation: **LSTM + MDN** over synthetic stroke trajectories derived from EMNIST

The repo supports training both models, generating samples, and running a lightweight comparison pipeline.

## Project Structure

```text
handwriting-compare/
  data/
  src/
    datasets/
      emnist.py
      emnist_to_strokes.py
    models/
      dcgan.py
      rnn_mdn.py
      classifier_cnn.py
    train_dcgan.py
    train_rnn.py
    train_classifier.py
    sample.py
    eval.py
    utils/
      io.py
      seed.py
      render.py
  tests/
    test_emnist_to_strokes.py
    test_model_shapes.py
    test_smoke_pipeline.py
  notebooks/
    comparison.ipynb
  README.md
  requirements.txt
```

## Environment Setup

Target runtime is **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training and Evaluation Commands

Run from repository root (`handwriting-compare/`).

1. Train realism classifier:

```bash
python -m src.train_classifier \
  --data-dir data \
  --out-dir runs/classifier \
  --epochs 12 \
  --batch-size 256 \
  --seed 42
```

2. Train DCGAN:

```bash
python -m src.train_dcgan \
  --data-dir data \
  --out-dir runs/dcgan \
  --epochs 40 \
  --batch-size 128 \
  --latent-dim 100 \
  --seed 42
```

3. Train sequence model (auto-builds stroke cache if missing):

```bash
python -m src.train_rnn \
  --strokes-path data/processed/emnist_letters_strokes_len160_seed42.npz \
  --data-dir data \
  --out-dir runs/rnn \
  --epochs 60 \
  --batch-size 128 \
  --mixtures 20 \
  --max-len 160 \
  --seed 42
```

4. Generate samples:

```bash
python -m src.sample \
  --model dcgan \
  --ckpt runs/dcgan/best.pt \
  --num-samples 64 \
  --out reports/samples_dcgan.png

python -m src.sample \
  --model rnn \
  --ckpt runs/rnn/best.pt \
  --num-samples 64 \
  --out reports/samples_rnn.png \
  --render
```

5. Evaluate comparison metrics:

```bash
python -m src.eval \
  --dcgan-ckpt runs/dcgan/best.pt \
  --rnn-ckpt runs/rnn/best.pt \
  --classifier-ckpt runs/classifier/best.pt \
  --num-samples 5000 \
  --out reports/metrics.json
```

## Expected Artifacts

- `runs/classifier/best.pt`: classifier checkpoint
- `runs/dcgan/best.pt`: GAN checkpoint
- `runs/rnn/best.pt`: sequence model checkpoint
- `reports/samples_dcgan.png`: GAN sample grid
- `reports/samples_dcgan_interp.png`: GAN latent interpolation strip
- `reports/samples_rnn.png`: rendered stroke sample grid
- `reports/samples_rnn_strokes.png`: raw stroke plots
- `reports/metrics.json`: comparison metrics
- `reports/metrics.csv`: CSV version of metrics

## Metrics in `reports/metrics.json`

- `gan_classifier_confidence_mean`
- `gan_classifier_confidence_p80`
- `gan_class_entropy`
- `rnn_stroke_length_mean`
- `rnn_pen_lifts_mean`
- `rnn_smoothness_mean_abs_turn`
- `rnn_render_classifier_confidence_mean`
- `rnn_render_class_entropy`

## Reproducibility

- Deterministic split file: `data/processed/splits_letters_seed42.npz`
- Cached stroke data: `data/processed/emnist_letters_strokes_len160_seed42.npz`
- Each run saves `config.json` plus metrics logs in its output directory.

## Notebook

Open `notebooks/comparison.ipynb` after running training/sample/eval to view side-by-side visuals and metric summary.

## Test Suite

```bash
pytest -q
```

Smoke tests execute CLI scripts in synthetic mode for quick validation without EMNIST download.

## Limitations

Stroke sequences are derived from image skeletons, so they approximate pen trajectories rather than true online handwriting capture.
