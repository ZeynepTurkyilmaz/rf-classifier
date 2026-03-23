from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../cpp'))
import fft_module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.cnn import SignalCNN

app = FastAPI(
    title="RF Sinyal Sınıflandırıcı",
    description="IQ verisinden sinyal tipini tahmin eden servis",
    version="1.0.0"
)

# Model başlarken bir kez yüklenir
_model  = None
_siniflar = None
_norm   = None

def modeli_yukle():
    global _model, _siniflar, _norm
    checkpoint = torch.load('model/signal_cnn.pt', map_location='cpu')
    _siniflar  = checkpoint['siniflar']
    num_classes = checkpoint['num_classes']
    _model     = SignalCNN(input_shape=checkpoint['input_shape'],
                           num_classes=num_classes)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()
    _norm = np.load('data/normalizasyon.npy')  # [mean, std]
    print("Model yüklendi:", _siniflar)

@app.on_event("startup")
def startup():
    modeli_yukle()

class IQVerisi(BaseModel):
    samples: list[float]  # [I0, Q0, I1, Q1, ...] formatında

@app.post("/classify")
def siniflandir(veri: IQVerisi):
    samples = np.array(veri.samples, dtype=np.float64)

    if len(samples) < 64:
        raise HTTPException(400, "En az 32 IQ çifti (64 değer) gerekli")

    # C++ FFT modülü ile spektrogram hesapla
    spec = fft_module.compute_spectrogram(samples, n_fft=32, hop_length=8)

    # Normalize et (eğitimle aynı normalizasyon)
    mean, std = _norm
    spec = (spec - mean) / (std + 1e-8)

    # PyTorch tensor: (1, 1, H, W)
    tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        cikti = _model(tensor)
        olasiliklar = torch.softmax(cikti, dim=1)[0].numpy()

    en_iyi_idx = int(olasiliklar.argmax())

    return {
        "sinyal_tipi":   _siniflar[en_iyi_idx],
        "guven":         float(olasiliklar[en_iyi_idx]),
        "tum_skorlar":   {s: float(p) for s, p in zip(_siniflar, olasiliklar)}
    }

@app.get("/health")
def saglik():
    return {"durum": "çalışıyor", "model": "signal_cnn.pt", "siniflar": _siniflar}
