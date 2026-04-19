import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample

TARGET_LEN = 16384

def bandpass(x, fs, low=None, high=None, order=4):
    nyq = 0.5 * fs
    if low is None: low = max(5, 0.01 * fs)
    if high is None: high = min(0.4 * fs, 5000)
    high = min(high, nyq - 1)
    low = max(low, 1)
    if low >= high: return x
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x)

def normalize(x):
    std = x.std()
    if std < 1e-8: return x - x.mean()
    return (x - x.mean()) / std

def preprocess_signal(sig, fs):
    sig = sig.reshape(-1)
    if len(sig) < 1000: return None, None
    orig_len = len(sig)
    if orig_len != TARGET_LEN:
        sig = resample(sig, TARGET_LEN)
        fs = fs * (TARGET_LEN / orig_len)
    sig = bandpass(sig, fs)
    sig = normalize(sig)
    return sig.astype(np.float32), fs

def envelope_spectrum(x, fs):
    env = np.abs(hilbert(x))
    spec = np.abs(np.fft.rfft(env)) ** 2
    spec = np.log1p(spec)
    freqs = np.fft.rfftfreq(len(env), 1 / fs)
    return freqs, spec

def rms(sig): return float(np.sqrt(np.mean(sig ** 2)))
def kurtosis(sig):
    mu, std = sig.mean(), sig.std() + 1e-8
    return float(np.mean(((sig - mu) / std) ** 4))

def crest_factor(sig):
    rms_val = rms(sig)
    peak = float(np.max(np.abs(sig)))
    return peak / (rms_val + 1e-8), peak

def signal_snr(sig):
    power = np.mean(sig ** 2)
    noise_est = np.median(np.abs(sig)) * 1.4826
    return float(max(0.0, 10 * np.log10(power / (noise_est ** 2 + 1e-8))))

def compute_defect_freqs(rpm_val, bpfi_mult, bpfo_mult, bsf_mult, ftf_mult):
    shaft_hz = rpm_val / 60.0
    return shaft_hz, shaft_hz * bpfi_mult, shaft_hz * bpfo_mult, shaft_hz * bsf_mult, shaft_hz * ftf_mult

def band_energy(spec, freqs, center_hz, bandwidth_hz):
    half_bw = bandwidth_hz / 2.0
    lo = max(center_hz - half_bw, 0.0)
    mask = (freqs >= lo) & (freqs <= center_hz + half_bw)
    if np.sum(mask) == 0: return 0.0
    return float(np.mean(spec[mask]))

def physics_features(sig, freqs, spec, rpm_val, bpfi_mult, bpfo_mult, bsf_mult, ftf_mult, campaign_progress=0.5):
    rms_val = np.log1p(rms(sig))
    kurt = kurtosis(sig)
    crest, peak = crest_factor(sig)
    crest = float(np.log1p(np.clip(crest, 0, 1e4)))
    peak = np.log1p(peak)
    snr = np.log1p(signal_snr(sig))

    shaft_hz, bpfi_hz, bpfo_hz, bsf_hz, ftf_hz = compute_defect_freqs(rpm_val, bpfi_mult, bpfo_mult, bsf_mult, ftf_mult)
    bw = 20.0
    e_bpfi = band_energy(spec, freqs, bpfi_hz, bw)
    e_bpfo = band_energy(spec, freqs, bpfo_hz, bw)
    e_bsf = band_energy(spec, freqs, bsf_hz, bw)

    total = e_bpfi + e_bpfo + e_bsf + 1e-6
    e_bpfi /= total; e_bpfo /= total; e_bsf /= total

    return np.array([
        rpm_val / 3000.0, rms_val, kurt, crest, peak, snr,
        shaft_hz / 100.0, e_bpfi, e_bpfo, e_bsf, campaign_progress
    ], dtype=np.float32)