import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_DIM = 64
N_HEADS = 4
N_MODALS = 2


# =========================
# LOAD SCD AUTOENCODER
# =========================
class SCDAnomalyModule(nn.Module):
    def __init__(self, model_path="./models/scd_anomaly.pth"):
        super().__init__()

        from src.models.scd_autoencoder import SCD_Autoencoder

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.model = SCD_Autoencoder()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False  # freeze

    def forward(self, scd):
        """
        scd: (B, H, W)
        returns anomaly score: (B, 1)
        """
        with torch.no_grad():
            err = self.model.reconstruction_error(scd)
        return err.unsqueeze(1)  # (B,1)


# =========================
# VIBRATION ENCODER
# =========================
class VibEncoder(nn.Module):
    def __init__(self, emb=FUSION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, emb)

    def forward(self, x):
        return self.proj(self.net(x).squeeze(-1))


# =========================
# PHYSICS ENCODER (UPDATED)
# =========================
class PhysEncoder(nn.Module):
    def __init__(self, in_dim, emb=FUSION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 32), nn.GELU(),   # +1 for anomaly score
            nn.Linear(32, emb),
        )

    def forward(self, p, anomaly_score):
        # concat anomaly score
        p = torch.cat([p, anomaly_score], dim=1)
        return self.net(p)


# =========================
# ATTENTION FUSION
# =========================
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, emb=FUSION_DIM, n_heads=N_HEADS, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, N_MODALS, emb) * 0.02)

        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.last_attn_weights = None

    def forward(self, z_vib, z_phys):
        tokens = torch.stack([z_vib, z_phys], dim=1) + self.pos_emb

        with torch.no_grad():
            _, w = self.attn_layer.self_attn(
                tokens, tokens, tokens,
                need_weights=True,
                average_attn_weights=True
            )
            self.last_attn_weights = w.detach()

        attended = self.attn_layer(tokens)
        return attended.flatten(start_dim=1)


# =========================
# FINAL CLASSIFIER (FUSED)
# =========================
class Classifier(nn.Module):
    def __init__(self, phys_dim, n_cls, emb=FUSION_DIM):
        super().__init__()

        self.vib = VibEncoder(emb)
        self.phys = PhysEncoder(phys_dim, emb)

        self.anomaly = SCDAnomalyModule()  # 🔥 NEW

        self.fusion = CrossModalAttentionFusion(emb)

        self.head = nn.Sequential(
            nn.LayerNorm(N_MODALS * emb),
            nn.Linear(N_MODALS * emb, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def forward(self, x, phys, scd):
        """
        x   : (B,1,L) vibration
        phys: (B, D)
        scd : (B, H, W)
        """

        z_vib = self.vib(x)

        # 🔥 anomaly score
        anomaly_score = self.anomaly(scd)

        z_phys = self.phys(phys, anomaly_score)

        fused = self.fusion(z_vib, z_phys)

        return self.head(fused)

    @property
    def attn_weights(self):
        return self.fusion.last_attn_weights


# =========================
# LOSS
# =========================
def focal_loss(logits, targets, alpha=0.25, gamma=2):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()