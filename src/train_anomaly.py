import torch
import numpy as np
import os
import joblib

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert, butter, filtfilt, stft

from src.data.loader import load_and_preprocess_data
from src.models.scd_autoencoder import SCD_Autoencoder


# =========================
# BANDPASS FILTER
# =========================
def bandpass_filter(X, fs=12000, low=500, high=5000, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq

    b, a = butter(order, [low, high], btype='band')

    if X.ndim == 3:
        X = X.mean(axis=1)

    return filtfilt(b, a, X, axis=1)


# =========================
# SCD FEATURES
# =========================
def compute_scd_features(X, fs=12000):
    X = bandpass_filter(X, fs)

    env = np.abs(hilbert(X, axis=1))

    _, _, Zxx = stft(env, fs=fs, nperseg=128, axis=1)

    spec = np.abs(Zxx)

    # normalize per sample (stable)
    spec = spec / (np.max(spec, axis=(1, 2), keepdims=True) + 1e-8)

    # log compression
    spec = np.log1p(spec)

    return spec


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    X_train, _, _, _, y_train, _ = load_and_preprocess_data("./data")

    # =========================
    # FEATURES
    # =========================
    P_train = compute_scd_features(X_train)
    P_train = P_train[:, :64, :64]

    print("Feature shape:", P_train.shape)

    # =========================
    # NORMAL DATA
    # =========================
    normal_mask = y_train == 0
    P_normal = P_train[normal_mask]

    print(f"Training on {len(P_normal)} normal samples")

    # =========================
    # SCALER
    # =========================
    scaler = StandardScaler()

    N, H, W = P_normal.shape

    P_normal_flat = scaler.fit_transform(P_normal.reshape(N, -1))
    P_train_flat = scaler.transform(P_train.reshape(P_train.shape[0], -1))

    P_normal = P_normal_flat.reshape(N, H, W)
    P_train = P_train_flat.reshape(P_train.shape[0], H, W)

    os.makedirs('./models', exist_ok=True)
    joblib.dump(scaler, './models/scaler.pkl')

    # =========================
    # MODEL
    # =========================
    model = SCD_Autoencoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # 🔥 better LR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    loader = DataLoader(
        torch.tensor(P_normal).float(),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # =========================
    # TRAIN
    # =========================
    for epoch in range(25):
        model.train()
        total_loss = 0

        # 🔥 decay noise over epochs
        noise_level = 0.01 * (1 - epoch / 25)

        for pb in loader:
            pb = pb.to(device)

            pb_noisy = pb + noise_level * torch.randn_like(pb)

            recon = model(pb_noisy)

            # SAFE ALIGN
            min_h = min(recon.shape[1], pb.shape[1])
            min_w = min(recon.shape[2], pb.shape[2])

            recon = recon[:, :min_h, :min_w]
            pb = pb[:, :min_h, :min_w]

            loss = torch.mean((recon - pb) ** 2)

            optimizer.zero_grad()
            loss.backward()

            # 🔥 gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/25 | Loss: {total_loss/len(loader):.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # =========================
    # ERROR ANALYSIS
    # =========================
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(
            torch.tensor(P_train).float().to(device)
        ).cpu().numpy()

    normal_errors = errors[y_train == 0]
    fault_errors = errors[y_train != 0]

    print("\n=== Training Stats ===")
    print(f"Normal Mean: {normal_errors.mean():.6f}")
    print(f"Normal Std : {normal_errors.std():.6f}")
    print(f"Fault  Mean: {fault_errors.mean():.6f}")

    # =========================
    # SAVE
    # =========================
    torch.save({
        'model_state_dict': model.state_dict(),
        'shape': (64, 64)
    }, './models/scd_anomaly.pth')

    print("Model saved!")


if __name__ == "__main__":
    main()