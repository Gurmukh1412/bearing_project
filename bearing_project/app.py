import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio

# 🔥 MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# rest of your imports (if any)
from src.fusion_model import Classifier
from src.models.scd_autoencoder import SCD_Autoencoder
from src.data.loader import load_and_preprocess_data
from scipy.signal import hilbert, butter, filtfilt, stft


# =========================
# 🔥 CRITICAL FIX: SIGNAL LENGTH
# =========================
def fix_signal_length(x, target_len=16384):
    x = np.squeeze(x)

    if len(x) > target_len:
        x = x[:target_len]
    elif len(x) < target_len:
        x = np.pad(x, (0, target_len - len(x)))

    return x


# =========================
# METRICS
# =========================
def confusion_matrix_np(y_true, y_pred, n_cls=4):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def compute_metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)

    precision, recall = [], []
    for c in range(4):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)

        precision.append(p)
        recall.append(r)

    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return acc, precision, recall, f1


# =========================
# SCD
# =========================
def compute_scd(x):
    nyq = 0.5 * 12000
    b, a = butter(4, [500/nyq, 5000/nyq], btype='band')

    x = np.squeeze(x)
    x = filtfilt(b, a, x)

    env = np.abs(hilbert(x))
    _, _, Zxx = stft(env, fs=12000, nperseg=128)

    spec = np.abs(Zxx)
    spec = spec / (np.max(spec) + 1e-8)
    spec = np.log1p(spec)

    return spec[:64, :64]


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    ckpt = torch.load("./models/fusion_model_best.pth", map_location=device)
    model = Classifier(phys_dim=ckpt['phys_dim'], n_cls=ckpt['n_cls'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ae_ckpt = torch.load("./models/scd_anomaly.pth", map_location=device)
    ae = SCD_Autoencoder()
    ae.load_state_dict(ae_ckpt['model_state_dict'])
    ae.eval()

    return model, ae


model, ae = load_models()


# =========================
# LOAD DATA
# =========================
X_train, X_test, P_train, P_test, y_train, y_test = load_and_preprocess_data("./data")


# =========================
# UI
# =========================
st.title("🏭 Bearing Fault Diagnosis Dashboard")

st.sidebar.header("📂 Input")

mode = st.sidebar.radio("Mode", ["Dataset (Cases)", "Upload .mat"])


# =========================
# DATA INPUT
# =========================
uploaded_signal = None

if mode == "Dataset (Cases)":
    case = st.sidebar.selectbox("Case (1–11)", list(range(1, 12)))
    indices = np.where((np.arange(len(X_test)) % 11) == (case - 1))[0]

    X_case = X_test[indices]
    P_case = P_test[indices]
    y_case = y_test[indices]

else:
    file = st.sidebar.file_uploader("Upload .mat", type=["mat"])
    if file is not None:
        mat = sio.loadmat(file)

        for key in mat:
            if isinstance(mat[key], np.ndarray):
                uploaded_signal = mat[key].squeeze()
                break


# =========================
# 🔍 SINGLE PREDICTION
# =========================
st.header("🔍 Prediction")

if mode == "Upload .mat" and uploaded_signal is not None:
    x = fix_signal_length(uploaded_signal)
    phys = np.zeros_like(P_test[0])

else:
    idx = st.slider("Sample", 0, len(X_case)-1, 0)
    x = fix_signal_length(X_case[idx])
    phys = P_case[idx]


scd = compute_scd(x)

# 🔥 CORRECT SHAPE (IMPORTANT)
x_t = torch.tensor(x).float().unsqueeze(0).unsqueeze(0)   # (1,1,L)
phys_t = torch.tensor(phys).float().unsqueeze(0)
scd_t = torch.tensor(scd).float().unsqueeze(0)

start = time.time()

with torch.no_grad():
    logits = model(x_t, phys_t, scd_t)
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred = np.argmax(probs)
    anomaly = ae.reconstruction_error(scd_t).item()

end = time.time()


# =========================
# DISPLAY
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediction", pred)
col2.metric("Confidence", f"{probs[pred]:.3f}")
col3.metric("Anomaly", f"{anomaly:.4f}")
col4.metric("Latency (ms)", f"{(end-start)*1000:.1f}")


# =========================
# VISUALS
# =========================
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(x)
    ax.set_title("Signal")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(scd, cmap='viridis', aspect='auto')
    ax.set_title("SCD")
    st.pyplot(fig)


# =========================
# DATASET PERFORMANCE
# =========================
if mode == "Dataset (Cases)":
    st.header("📊 Case Performance")

    preds = []

    for i in range(len(X_case)):
        x = fix_signal_length(X_case[i])
        phys = P_case[i]
        scd = compute_scd(x)

        x_t = torch.tensor(x).float().unsqueeze(0).unsqueeze(0)
        phys_t = torch.tensor(phys).float().unsqueeze(0)
        scd_t = torch.tensor(scd).float().unsqueeze(0)

        with torch.no_grad():
            logits = model(x_t, phys_t, scd_t)
            preds.append(torch.argmax(logits).item())

    preds = np.array(preds)

    acc, prec, rec, f1 = compute_metrics(y_case, preds)
    cm = confusion_matrix_np(y_case, preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1", f"{f1:.3f}")

    st.write("Confusion Matrix")
    st.write(cm)