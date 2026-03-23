import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # WSL'de ekran olmadığı için
import matplotlib.pyplot as plt

# Veriyi yükle
with open('data/RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Veri yapısını anla
print("Toplam kayıt sayısı:", len(data))
print("\nAnahtar formatı: (sinyal_tipi, SNR_değeri)")
print("İlk 5 anahtar:", list(data.keys())[:5])

# Bir örnek incele
key = list(data.keys())[0]
sample = data[key]
print(f"\n'{key}' için örnek boyutu: {sample.shape}")
print("Her örnek: (128 örnek, 2 kanal [I, Q])")

# Kaç sinyal tipi var?
sinyal_tipleri = list(set([k[0] for k in data.keys()]))
snr_degerleri  = sorted(list(set([k[1] for k in data.keys()])))
print("\nSinyal tipleri:", sinyal_tipleri)
print("SNR değerleri:", snr_degerleri)

# Veriyi numpy array'e dönüştür
X = []
Y = []
sinif_listesi = sorted(sinyal_tipleri)

for (mod, snr), samples in data.items():
    if snr >= 0:  # Düşük SNR'li gürültülü örnekleri şimdilik atla
        label = sinif_listesi.index(mod)
        for s in samples:
            X.append(s)       # shape: (128, 2)
            Y.append(label)

X = np.array(X)  # (N, 128, 2)
X = X.transpose(0, 2, 1)
Y = np.array(Y)  # (N,)
print(f"\nSNR >= 0 koşulunda toplam örnek: {len(X)}")
print("Sınıflar:", sinif_listesi)

# Bir örneği görselleştir
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(X[0, :, 0], label='I kanalı')
axes[0].plot(X[0, :, 1], label='Q kanalı')
axes[0].set_title(f'Zaman alanı — {sinif_listesi[Y[0]]}')
axes[0].legend()

# Spektrogram oluştur (şimdilik Python ile, sonra C++ ile yapacağız)
from scipy import signal
f, t, Sxx = signal.spectrogram(X[0, :, 0] + 1j * X[0, :, 1],
                                 fs=1.0, nperseg=16, noverlap=8)
axes[1].pcolormesh(t, f, 10 * np.log10(np.abs(Sxx) + 1e-10))
axes[1].set_title('Spektrogram (dB)')
axes[1].set_ylabel('Frekans')
axes[1].set_xlabel('Zaman')

plt.tight_layout()
plt.savefig('data/ornek_spektrogram.png', dpi=150)
print("\nGörsel kaydedildi: data/ornek_spektrogram.png")
