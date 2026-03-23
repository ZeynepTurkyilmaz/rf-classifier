import pickle
import numpy as np
import sys
import os

# C++ modülünü import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../cpp'))
import fft_module

def iq_to_spectrogram(iq_sample, n_fft=32, hop_length=8):
    """
    iq_sample: numpy array, shape (128, 2) — RadioML formatı
    Döndürür: 2D spektrogram array
    """
    # RadioML formatı (128, 2) → FFTW beklentisi [I0, Q0, I1, Q1, ...]
    interleaved = iq_sample.flatten().astype(np.float64)
    spec = fft_module.compute_spectrogram(interleaved, n_fft=n_fft, hop_length=hop_length)
    return spec

def hazirla_dataset(pkl_yolu, snr_min=0, n_fft=32, hop_length=8):
    print("Veri yükleniyor...")
    with open(pkl_yolu, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    sinif_listesi = sorted(list(set([k[0] for k in data.keys()])))
    print(f"Sınıflar ({len(sinif_listesi)} adet):", sinif_listesi)

    X_list, Y_list = [], []
    toplam = 0

    for (mod, snr), samples in data.items():
        if snr < snr_min:
            continue
        label = sinif_listesi.index(mod)
        for sample in samples:
            spec = iq_to_spectrogram(sample, n_fft, hop_length)
            X_list.append(spec)
            Y_list.append(label)
            toplam += 1

    X = np.array(X_list, dtype=np.float32)  # (N, zaman_adımı, frekans_bin)
    Y = np.array(Y_list, dtype=np.int64)

    # CNN için kanal boyutu ekle: (N, 1, H, W)
    X = X[:, np.newaxis, :, :]

    print(f"\nToplam örnek: {toplam}")
    print(f"X shape: {X.shape}  — (örnek_sayısı, kanal, zaman, frekans)")
    print(f"Y shape: {Y.shape}")
    print(f"Değer aralığı: min={X.min():.1f} dB, max={X.max():.1f} dB")

    # Normalize et: her spektrogramı sıfır ortalama, birim varyansa getir
    mean = X.mean()
    std  = X.std()
    X = (X - mean) / (std + 1e-8)

    # Kaydet
    np.save('data/X_train.npy', X)
    np.save('data/Y_train.npy', Y)
    np.save('data/sinif_listesi.npy', np.array(sinif_listesi))
    np.save('data/normalizasyon.npy', np.array([mean, std]))

    print("\nDosyalar kaydedildi: data/X_train.npy, data/Y_train.npy")
    return X, Y, sinif_listesi

if __name__ == '__main__':
    X, Y, siniflar = hazirla_dataset('data/RML2016.10a_dict.pkl')
