import torch
import torch.nn as nn


class SCD_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================
        # ENCODER
        # =========================
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # =========================
        # DECODER
        # =========================
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Input: (B, H, W)
        x = x.unsqueeze(1)  # → (B,1,H,W)

        z = self.encoder(x)
        out = self.decoder(z)

        return out.squeeze(1)  # → (B,H,W)

    def reconstruction_error(self, x):
        recon = self.forward(x)

        # =========================
        # SAFE SHAPE ALIGNMENT
        # =========================
        min_h = min(recon.shape[1], x.shape[1])
        min_w = min(recon.shape[2], x.shape[2])

        recon = recon[:, :min_h, :min_w]
        x = x[:, :min_h, :min_w]

        return torch.mean((x - recon) ** 2, dim=(1, 2))