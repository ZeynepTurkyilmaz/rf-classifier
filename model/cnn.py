import torch
import torch.nn as nn

class SignalCNN(nn.Module):
    """
    RF sinyal spektrogramlarını sınıflandıran CNN.
    Giriş: (batch, 1, yükseklik, genişlik) — tek kanallı spektrogram
    Çıkış: (batch, num_classes) — her sınıf için ham skor
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # Evrişim bloklarını tanımla
        self.features = nn.Sequential(
            # Blok 1: küçük özellikleri yakala
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # boyutu yarıya indir
            nn.Dropout2d(0.25),

            # Blok 2: orta ölçekli özellikleri yakala
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Blok 3: yüksek seviye özellikleri
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # ne kadar küçük veri olursa olsun 4x4'e indir
        )

        # Tam bağlantılı katmanlar
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
