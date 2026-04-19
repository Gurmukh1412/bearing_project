import torch
import torch.nn as nn


class TransformerAE(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=2, num_layers=1):
        super().__init__()

        # Project input → latent dimension
        self.input_proj = nn.Linear(n_features, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_features)
        )

    def forward(self, x):
        # x: (B, F)
        x = self.input_proj(x)   # (B, d_model)
        x = x.unsqueeze(1)      # (B, 1, d_model)

        z = self.encoder(x)     # (B, 1, d_model)
        z = z.squeeze(1)        # (B, d_model)

        out = self.decoder(z)
        return out

    def reconstruction_error(self, x):
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)
    





import torch
import torch.nn as nn


class Conv1DAE(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,F)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.squeeze(1)

    def reconstruction_error(self, x):
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)
    

def reconstruction_error(self, x):
    recon = self.forward(x)

    # FIX: align shapes
    if recon.shape[1] != x.shape[1]:
        min_len = min(recon.shape[1], x.shape[1])
        recon = recon[:, :min_len]
        x = x[:, :min_len]

    return torch.mean((x - recon) ** 2, dim=1)