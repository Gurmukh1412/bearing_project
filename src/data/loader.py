import os
import glob
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from src.utils.physics import preprocess_signal, envelope_spectrum, physics_features

def get_struct_field(sample_struct, field_name):
    if field_name not in sample_struct.dtype.names: return None
    val = sample_struct[field_name]
    if isinstance(val, np.ndarray) and val.dtype == object:
        if val.size > 0:
            item = val.flatten()[0]
            if isinstance(item, np.ndarray): return item
            try: return np.array(item)
            except: return item
    return val

def find_main_struct(mat_content):
    for key in ['train', 'DS', 'FS', 'Upper', 'data']:
        if key in mat_content:
            val = mat_content[key]
            if isinstance(val, np.ndarray) and val.dtype.names is not None:
                if 'rawData' in val.dtype.names or 'label' in val.dtype.names:
                    return key
    return None

def load_and_preprocess_data(data_root, seq_len=16384, test_size=0.2):
    all_signals, all_phys, all_labels, all_rpms = [], [], [], []
    
    case_folders = sorted(glob.glob(os.path.join(data_root, "Case_*")))
    if not case_folders:
        case_folders = sorted([f for f in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(f)])

    print(f"Found {len(case_folders)} case folders.")

    for case_path in case_folders:
        mat_files = glob.glob(os.path.join(case_path, "*.mat"))
        if not mat_files: continue
        
        for mat_file in mat_files:
            try:
                mat_content = loadmat(mat_file)
                struct_name = find_main_struct(mat_content)
                if struct_name is None: continue
                
                data_struct = mat_content[struct_name]
                if data_struct.shape[0] == 0: continue
                sample = data_struct[0, 0]
                
                raw_data_matrix = get_struct_field(sample, 'rawData')
                labels_arr = get_struct_field(sample, 'label')
                rpm_val = get_struct_field(sample, 'RPM')
                
                if raw_data_matrix is None or labels_arr is None: continue
                
                if isinstance(rpm_val, np.ndarray): rpm_val = float(rpm_val.flatten()[0])
                else: rpm_val = 1797.0
                
                labels_arr = np.asarray(labels_arr).flatten()
                if raw_data_matrix.ndim == 1: raw_data_matrix = raw_data_matrix.reshape(1, -1)
                
                n_samples = raw_data_matrix.shape[0]
                for i in range(n_samples):
                    sig = raw_data_matrix[i, :]
                    if i >= len(labels_arr): continue
                    
                    lbl = int(labels_arr[i])
                    if lbl == -1: continue
                    
                    sig_proc, fs_proc = preprocess_signal(sig, 25600) # Assume 25.6kHz or extract from metadata
                    if sig_proc is None: continue
                    
                    freqs, spec = envelope_spectrum(sig_proc, fs_proc)
                    phys_feat = physics_features(sig_proc, freqs, spec, rpm_val, 
                                                 bpfi_mult=5.415, bpfo_mult=3.585, 
                                                 bsf_mult=2.357, ftf_mult=0.398)
                    
                    all_signals.append(sig_proc)
                    all_phys.append(phys_feat)
                    all_labels.append(lbl)
                    all_rpms.append(rpm_val)
                    
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")

    if not all_signals: raise ValueError("No valid data loaded.")
    
    X_np = np.array(all_signals)[:, np.newaxis, :] # (N, 1, L)
    PHYS_np = np.array(all_phys)                   # (N, D)
    y_np = np.array(all_labels)
    
    print(f"Loaded {len(y_np)} samples. Dist: Healthy={np.sum(y_np==0)}, Faulty={np.sum(y_np!=0)}")
    
    X_train, X_test, P_train, P_test, y_train, y_test = train_test_split(
        X_np, PHYS_np, y_np, test_size=test_size, stratify=y_np, random_state=42
    )
    
    return X_train, X_test, P_train, P_test, y_train, y_test