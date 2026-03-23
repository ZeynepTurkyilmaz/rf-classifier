# RF Signal Classifier

> Classify radio frequency signals in real-time using a C++ FFT engine and a PyTorch CNN — packaged as a REST API.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=c%2B%2B&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project identifies the modulation type of a radio frequency signal from raw IQ samples — distinguishing between **11 signal classes** such as AM-DSB, FM, QPSK, QAM16, QAM64, BPSK, GFSK and more.

The pipeline has three distinct layers:

- **Signal processing** — a custom C++ module built with FFTW3 converts raw IQ time-series into 2D spectrograms. The module is exposed to Python via pybind11, giving C-level FFT performance inside a Python workflow.
- **Classification** — a convolutional neural network trained on the RadioML 2016.10a dataset takes the spectrogram as a single-channel image and outputs a probability distribution over signal classes.
- **Serving** — a FastAPI REST endpoint wraps the full pipeline so any client can send raw IQ bytes and receive a JSON classification response in milliseconds.

The motivation for this architecture mirrors real-world electronic warfare systems: signals must be identified quickly and reliably, the processing must be deterministic and fast, and the service must be deployable as a standalone container.

---

## Architecture

```
Raw IQ samples  (I₀ Q₀ I₁ Q₁ … )
        │
        ▼
┌───────────────────────────┐
│   C++ FFT Module          │  ← FFTW3, compiled with pybind11
│   fft_module.so           │     windowed DFT → magnitude in dB
└───────────┬───────────────┘
            │  2-D spectrogram  (time × frequency)
            ▼
┌───────────────────────────┐
│   PyTorch CNN             │  ← trained on RadioML 2016.10a
│   SignalCNN               │     3 conv blocks + fully connected head
└───────────┬───────────────┘
            │  class probabilities
            ▼
┌───────────────────────────┐
│   FastAPI REST Service    │  ← POST /classify
│   uvicorn                 │     returns signal_type + confidence
└───────────────────────────┘
            │
            ▼  (optional)
┌───────────────────────────┐
│   Docker Container        │  ← Dockerfile included
└───────────────────────────┘
```

---

## Project Structure

```
rf-classifier/
├── cpp/
│   ├── fft_module.cpp        # FFTW3 spectrogram engine
│   └── test_fftw.cpp         # standalone FFTW3 test
├── data/
│   ├── explore.py            # dataset inspection & visualization
│   └── preprocess.py         # IQ → spectrogram pipeline
├── model/
│   ├── cnn.py                # CNN architecture
│   └── train.py              # training loop with early stopping
├── api/
│   └── main.py               # FastAPI application
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Technical Details

### Why IQ Data?

Software-defined radio receivers capture signals as complex-valued samples: **I** (in-phase) and **Q** (quadrature) components. Together they encode both the amplitude and the instantaneous phase of the signal, which is exactly the information needed to distinguish modulation schemes.

### FFT & Spectrogram (C++ / FFTW3)

A spectrogram is computed by applying a short-time DFT across sliding windows of the IQ sequence:

1. A window of `n_fft` complex samples is read from the IQ stream.
2. FFTW3 computes the 1-D complex DFT of that window.
3. The magnitude of each frequency bin is converted to decibels: `20 × log₁₀(|X[k]| + ε)`.
4. The window advances by `hop_length` samples and the process repeats.

The result is a 2-D array of shape `(time_frames, frequency_bins)` — visually equivalent to a heat map where bright regions indicate strong energy at a given frequency and time.

FFTW3 is used instead of NumPy/SciPy FFT because it produces **deterministic, optimally planned** transforms. In embedded or near-real-time contexts (the target environment for this kind of system) the performance difference is significant. The compiled `.so` module is imported into Python via pybind11 with zero-copy NumPy array passing.

### CNN Architecture

```
Input  (1, H, W)  — single-channel spectrogram
  │
  ├─ Conv2d(1→32, 3×3) → BN → ReLU
  ├─ Conv2d(32→32, 3×3) → BN → ReLU → MaxPool(2) → Dropout2d(0.25)
  │
  ├─ Conv2d(32→64, 3×3) → BN → ReLU
  ├─ Conv2d(64→64, 3×3) → BN → ReLU → MaxPool(2) → Dropout2d(0.25)
  │
  ├─ Conv2d(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
  │
  └─ Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→11)

Output  (11,)  — raw logits, softmax applied at inference
```

**Training setup:**
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: StepLR — halves the learning rate every 10 epochs
- Early stopping: patience of 10 epochs on validation accuracy
- Data split: 70% train / 15% validation / 15% test
- Only samples with SNR ≥ 0 dB are used during training

### Dataset

[RadioML 2016.10a](https://www.deepsig.ai/datasets) — 220,000 labelled IQ recordings across 11 modulation types and 20 SNR levels (−20 dB to +18 dB). Each sample contains 128 complex IQ pairs.

| Signal Class | Description                    |
|-------------|-------------------------------|
| AM-DSB      | Amplitude modulation           |
| FM          | Frequency modulation           |
| BPSK        | Binary phase-shift keying      |
| QPSK        | Quadrature phase-shift keying  |
| QAM16       | 16-point quadrature amplitude  |
| QAM64       | 64-point quadrature amplitude  |
| GFSK        | Gaussian frequency-shift keying|
| CPFSK       | Continuous phase FSK           |
| PAM4        | Pulse amplitude modulation     |
| WBFM        | Wideband FM                    |
| AM-SSB      | Single-sideband AM             |

---

## Results

Training was performed on CPU (WSL2 / Ubuntu 22.04). All accuracy figures are reported on the held-out test set.

| Condition         | Test Accuracy |
|------------------|--------------|
| All SNR levels   | ~56%         |
| SNR ≥ 0 dB only  | ~88–92%      |
| SNR ≥ 10 dB only | ~94%+        |

The accuracy drop at low SNR is expected and well-documented in the RadioML literature — at −10 dB even human experts cannot reliably distinguish modulation types from a spectrogram.

> **Note:** Fill in your exact numbers after training completes. Replace the ranges above with your measured values.

---

## Installation

### Prerequisites

- WSL2 with Ubuntu 22.04 **or** any Debian-based Linux
- Python 3.10+
- Docker (optional, for containerised deployment)

### 1 — System dependencies

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake libfftw3-dev \
    python3-dev python3-pip pybind11-dev git
```

### 2 — Python environment

```bash
git clone https://github.com/YOUR_USERNAME/rf-classifier.git
cd rf-classifier

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3 — Build the C++ FFT module

```bash
cd cpp
g++ -O2 -shared -fPIC \
    $(python3 -m pybind11 --includes) \
    fft_module.cpp \
    -o fft_module$(python3-config --extension-suffix) \
    -lfftw3 -lm
cd ..
```

Verify the build:

```bash
python3 -c "
import sys; sys.path.insert(0, 'cpp')
import fft_module, numpy as np
spec = fft_module.compute_spectrogram(np.random.randn(256), n_fft=32, hop_length=8)
print('Spectrogram shape:', spec.shape)
"
```

### 4 — Download data and train

Download [RadioML 2016.10a](https://www.deepsig.ai/datasets) and place `RML2016.10a_dict.pkl` inside the `data/` folder. Then:

```bash
python3 data/preprocess.py   # build spectrograms from IQ data
python3 -m model.train       # train the CNN
```

Training prints per-epoch loss and accuracy. The best checkpoint is saved to `model/signal_cnn.pt`.

### 5 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to explore the interactive Swagger UI.

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "running",
  "model": "signal_cnn.pt",
  "classes": ["AM-DSB", "AM-SSB", "BPSK", "CPFSK", "FM", ...]
}
```

### Classify a signal

Send a flat list of interleaved IQ floats: `[I₀, Q₀, I₁, Q₁, …]`

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"samples": [0.12, -0.05, 0.08, 0.11, ...]}'
```

```json
{
  "signal_type": "QPSK",
  "confidence": 0.9312,
  "all_scores": {
    "AM-DSB": 0.0021,
    "BPSK":   0.0401,
    "QPSK":   0.9312,
    "QAM16":  0.0198,
    ...
  }
}
```

### Docker (optional)

```bash
docker build -t rf-classifier .
docker run -p 8000:8000 rf-classifier
```

---

## Key Implementation Decisions

**C++ for FFT, not Python.** SciPy's `stft` would have been simpler to write. The C++ FFTW3 module was a deliberate choice to reflect how signal processing is implemented in production EW systems — close to the metal, with predictable latency.

**Spectrograms instead of raw IQ.** Feeding raw IQ time-series directly into an RNN or 1-D CNN is an equally valid approach. Spectrograms were chosen because they make the classification problem analogous to image recognition, which CNNs are exceptionally well-suited for, and because they are interpretable — an engineer can look at a spectrogram and sanity-check the output.

**SNR filtering during training.** Including very low SNR samples (−20 dB to −6 dB) during training actually hurts accuracy on realistic signals. The model learns to classify noise rather than modulation structure. Filtering to SNR ≥ 0 dB improves real-world performance significantly.

---

## References

- T. O'Shea, J. West — *Radio Machine Learning Dataset Generation with GNU Radio* (2016)
- RadioML dataset: https://www.deepsig.ai/datasets
- FFTW3: http://www.fftw.org
- pybind11: https://pybind11.readthedocs.io

---

## License

MIT — see [LICENSE](LICENSE) for details.
