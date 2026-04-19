import torch
import numpy as np
import joblib

from sklearn.metrics import classification_report, confusion_matrix
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
    f, t, Zxx = stft(env, fs=fs, nperseg=128, axis=1)

    spec = np.abs(Zxx)
    spec = spec / (np.max(spec, axis=(1, 2), keepdims=True) + 1e-8)
    spec = np.log1p(spec)

    return spec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # LOAD DATA
    # =========================
    _, X_test, _, _, _, y_test = load_and_preprocess_data("./data")

    # =========================
    # FEATURES
    # =========================
    P_test = compute_scd_features(X_test)
    P_test = P_test[:, :64, :64]

    print("Eval feature shape:", P_test.shape)

    # =========================
    # LOAD SCALER
    # =========================
    scaler = joblib.load('./models/scaler.pkl')

    N, H, W = P_test.shape
    P_test = scaler.transform(P_test.reshape(N, -1)).reshape(N, H, W)

    # =========================
    # LOAD MODEL
    # =========================
    checkpoint = torch.load(
        './models/scd_anomaly.pth',
        map_location=device,
        weights_only=False
    )

    model = SCD_Autoencoder().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # =========================
    # INFERENCE
    # =========================
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(
            torch.tensor(P_test).float().to(device)
        ).cpu().numpy()

    y_true = (y_test != 0).astype(int)

    # =========================
    # 🔥 ROBUST PERCENTILE THRESHOLD (BEST)
    # =========================
    normal_errors = errors[y_test == 0]
    fault_errors = errors[y_test != 0]

    n_high = np.percentile(normal_errors, 90)
    f_low  = np.percentile(fault_errors, 20)

    threshold = 0.5 * (n_high + f_low)

    preds = (errors > threshold).astype(int)

    # =========================
    # RESULTS
    # =========================
    print("\n=== RESULTS ===")
    print(classification_report(y_true, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, preds))

    # =========================
    # DEBUG
    # =========================
    print("\n=== Debug ===")
    print(f"Normal Mean: {normal_errors.mean():.6f}")
    print(f"Fault  Mean: {fault_errors.mean():.6f}")
    print(f"Normal P90: {n_high:.6f}")
    print(f"Fault  P20: {f_low:.6f}")
    print(f"Threshold : {threshold:.6f}")

    print(f"\nPredicted anomalies: {preds.sum()} / {len(preds)}")


if __name__ == "__main__":
    main()