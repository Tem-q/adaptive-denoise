import torch
import torch.nn as nn


class _LoadMixin:
    def load(self, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state)
        self.to(device)
        self.eval()
        return self


class UNet(_LoadMixin, nn.Module):
    """Three-level U-Net for 128x128 RGB patches."""

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256)):
        super().__init__()

        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = self._block(features[2], features[2] * 2)

        self.upconv3 = nn.ConvTranspose2d(features[2] * 2, features[2], 2, 2)
        self.decoder3 = self._block(features[2] * 2, features[2])
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.decoder2 = self._block(features[1] * 2, features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.decoder1 = self._block(features[0] * 2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.final_conv(dec1)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class NoiseClassifier(_LoadMixin, nn.Module):
    """Noise-type classifier for 128x128 RGB patches.

    Classes: 0 = real, 1 = salt_pepper, 2 = speckle.
    """

    NOISE_TYPES = ("real", "salt_pepper", "speckle")

    def __init__(self, in_channels=3, num_classes=3, features=(32, 64, 128, 256)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(features[0], features[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(features[1], features[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[2], features[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(features[2], features[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(features[3], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_noise_type(self, x, device="cpu"):
        """Predict noise-type indices and names for a batch.

        Returns:
            (Tensor[B], List[str]): class indices and corresponding names.
        """
        self.eval()
        self.to(device)

        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)

        logits = self(x)
        idx = torch.argmax(logits, dim=1).cpu()
        names = [self.NOISE_TYPES[i] for i in idx.numpy()]
        return idx, names
