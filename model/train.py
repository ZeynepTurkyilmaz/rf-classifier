import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys, os

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from model.cnn import SignalCNN

def egit():
    # --- Veriyi yükle ---
    print("Veri yükleniyor...")
    X = np.load('data/X_train.npy')
    Y = np.load('data/Y_train.npy')
    siniflar = np.load('data/sinif_listesi.npy')
    num_classes = len(siniflar)

    print(f"Veri: {X.shape}, Sınıf sayısı: {num_classes}")

    # NumPy → PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, Y_tensor)

    # %70 eğitim, %15 doğrulama, %15 test
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    print(f"Eğitim: {n_train}, Doğrulama: {n_val}, Test: {n_test}")

    # --- Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    model = SignalCNN(input_shape=X.shape[1:], num_classes=num_classes).to(device)

    kayip_fn  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # --- Eğitim döngüsü ---
    en_iyi_val_kayip = float('inf')
    en_iyi_dogruluk  = 0.0
    sabir = 0
    maks_sabir = 10  # 10 epoch iyileşme olmazsa dur (early stopping)

    for epoch in range(1, 51):  # maksimum 50 epoch
        # Eğitim
        model.train()
        toplam_kayip, dogru, toplam = 0, 0, 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            cikti   = model(X_batch)
            kayip   = kayip_fn(cikti, Y_batch)
            kayip.backward()
            optimizer.step()

            toplam_kayip += kayip.item() * len(Y_batch)
            dogru        += (cikti.argmax(1) == Y_batch).sum().item()
            toplam       += len(Y_batch)

        train_kayip = toplam_kayip / toplam
        train_acc   = dogru / toplam

        # Doğrulama
        model.eval()
        val_kayip, val_dogru, val_toplam = 0, 0, 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                cikti     = model(X_batch)
                kayip     = kayip_fn(cikti, Y_batch)
                val_kayip += kayip.item() * len(Y_batch)
                val_dogru += (cikti.argmax(1) == Y_batch).sum().item()
                val_toplam += len(Y_batch)

        val_kayip /= val_toplam
        val_acc    = val_dogru / val_toplam

        scheduler.step()

        print(f"Epoch {epoch:2d} | "
              f"Eğitim: kayıp={train_kayip:.4f} acc={train_acc:.3f} | "
              f"Doğrulama: kayıp={val_kayip:.4f} acc={val_acc:.3f}")

        # En iyi modeli kaydet
        if val_acc > en_iyi_dogruluk:
            en_iyi_dogruluk = val_acc
            sabir = 0
            torch.save({
                'model_state': model.state_dict(),
                'siniflar':    siniflar.tolist(),
                'input_shape': X.shape[1:],
                'num_classes': num_classes,
                'val_acc':     val_acc
            }, 'model/signal_cnn.pt')
            print(f"  → En iyi model kaydedildi (val_acc={val_acc:.3f})")
        else:
            sabir += 1
            if sabir >= maks_sabir:
                print(f"\nErken durdurma: {maks_sabir} epoch iyileşme olmadı.")
                break

    # --- Test ---
    print("\n--- Test Sonuçları ---")
    checkpoint = torch.load('model/signal_cnn.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    test_dogru, test_toplam = 0, 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            cikti       = model(X_batch)
            test_dogru  += (cikti.argmax(1) == Y_batch).sum().item()
            test_toplam += len(Y_batch)

    test_acc = test_dogru / test_toplam
    print(f"Test doğruluğu: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Sınıflar: {siniflar.tolist()}")

if __name__ == '__main__':
    egit()
