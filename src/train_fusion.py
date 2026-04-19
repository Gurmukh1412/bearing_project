import torch
import numpy as np
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.fusion_model import Classifier, focal_loss
from src.data.loader import load_and_preprocess_data

from scipy.signal import hilbert, butter, filtfilt, stft


# =========================
# MEMORY-SAFE SCD FEATURES
# =========================
def compute_scd_features_batch(X, batch_size=128, fs=12000):
    outputs = []

    nyq = 0.5 * fs
    b, a = butter(4, [500/nyq, 5000/nyq], btype='band')

    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]

        if xb.ndim == 3:
            xb = xb.mean(axis=1)

        xb = filtfilt(b, a, xb, axis=1)
        env = np.abs(hilbert(xb, axis=1))
        _, _, Zxx = stft(env, fs=fs, nperseg=128, axis=1)

        spec = np.abs(Zxx)
        spec = spec / (np.max(spec, axis=(1, 2), keepdims=True) + 1e-8)
        spec = np.log1p(spec)

        outputs.append(spec[:, :64, :64])

    return np.concatenate(outputs, axis=0)


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs("./models", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================
    X_train, X_test, P_train, P_test, y_train, y_test = load_and_preprocess_data("./data")

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # =========================
    # PREP INPUTS
    # =========================
    X_train = torch.tensor(X_train).float()
    X_test  = torch.tensor(X_test).float()

    P_train = torch.tensor(P_train).float()
    P_test  = torch.tensor(P_test).float()

    y_train = torch.tensor(y_train).long()
    y_test  = torch.tensor(y_test).long()

    # =========================
    # SCD FEATURES
    # =========================
    print("Computing SCD features (train)...")
    scd_train = torch.tensor(
        compute_scd_features_batch(X_train.squeeze(1).numpy(), batch_size=128)
    ).float()

    print("Computing SCD features (test)...")
    scd_test = torch.tensor(
        compute_scd_features_batch(X_test.squeeze(1).numpy(), batch_size=128)
    ).float()

    print("SCD shape:", scd_train.shape)

    # =========================
    # DATALOADER
    # =========================
    train_loader = DataLoader(
        TensorDataset(X_train, P_train, scd_train, y_train),
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, P_test, scd_test, y_test),
        batch_size=64,
        shuffle=False
    )

    # =========================
    # MODEL
    # =========================
    model = Classifier(
        phys_dim=P_train.shape[1],
        n_cls=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0

    # =========================
    # TRAIN
    # =========================
    print("\nTraining Fusion Model...\n")

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, phys, scd, y in train_loader:
            x, phys, scd, y = x.to(device), phys.to(device), scd.to(device), y.to(device)

            logits = model(x, phys, scd)
            loss = focal_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f}")

        # =========================
        # QUICK VALIDATION (FOR BEST MODEL)
        # =========================
        model.eval()
        preds_tmp = []
        true_tmp = []

        with torch.no_grad():
            for x, phys, scd, y in test_loader:
                x, phys, scd = x.to(device), phys.to(device), scd.to(device)
                out = model(x, phys, scd)
                p = torch.argmax(out, dim=1).cpu().numpy()

                preds_tmp.extend(p)
                true_tmp.extend(y.numpy())

        acc_tmp = accuracy_score(true_tmp, preds_tmp)
        print(f"Validation Acc: {acc_tmp:.4f}")

        # save best
        if acc_tmp > best_acc:
            best_acc = acc_tmp
            torch.save({
                'model_state_dict': model.state_dict(),
                'phys_dim': P_train.shape[1],
                'n_cls': 4
            }, "./models/fusion_model_best.pth")

            print("✅ Saved BEST model")

    # =========================
    # FINAL SAVE
    # =========================
    torch.save({
        'model_state_dict': model.state_dict(),
        'phys_dim': P_train.shape[1],
        'n_cls': 4
    }, "./models/fusion_model_last.pth")

    print("💾 Saved FINAL model")

    # =========================
    # FINAL EVALUATION
    # =========================
    print("\nEvaluating...\n")

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, phys, scd, y in test_loader:
            x, phys, scd = x.to(device), phys.to(device), scd.to(device)

            logits = model(x, phys, scd)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y.numpy())

    acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    print("\n=== FUSION MODEL RESULTS ===")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds))


if __name__ == "__main__":
    main()