import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from src.data.loader import load_and_preprocess_data
from src.fusion_model import Classifier, focal_loss
from src.utils.plots import run_evaluation

def make_weighted_sampler(y):
    counts = np.bincount(y)
    class_w = 1.0 / counts.astype(float)
    sample_w = class_w[y]
    return WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float), num_samples=len(sample_w), replacement=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # Load Data
    X_train, X_test, P_train, P_test, y_train, y_test = load_and_preprocess_data("./data")
    
    # Remap Labels to 0..N-1
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    inv_map = {v: k for k, v in label_map.items()}
    
    y_train_remapped = np.vectorize(label_map.get)(y_train)
    y_test_remapped = np.vectorize(label_map.get)(y_test)
    n_cls = len(label_map)
    
    print(f"Classes: {label_map}")
    
    # Tensors
    X_tr, P_tr, y_tr = torch.tensor(X_train).float(), torch.tensor(P_train).float(), torch.tensor(y_train_remapped).long()
    X_te, P_te, y_te = torch.tensor(X_test).float(), torch.tensor(P_test).float(), torch.tensor(y_test_remapped).long()
    
    sampler = make_weighted_sampler(y_train_remapped)
    train_loader = DataLoader(TensorDataset(X_tr, P_tr, y_tr), batch_size=32, sampler=sampler)
    test_loader = DataLoader(TensorDataset(X_te, P_te, y_te), batch_size=32)
    
    model = Classifier(phys_dim=P_train.shape[1], n_cls=n_cls).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"Training Classifier with {n_cls} classes...")
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        for xb, pb, yb in train_loader:
            xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)
            logits = model(xb, pb)
            loss = focal_loss(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20 | Loss: {total_loss/len(train_loader):.4f}")
        
    # Evaluate
    run_evaluation(model, test_loader, inv_map, label_map, out_dir="./outputs")

if __name__ == "__main__":
    main()