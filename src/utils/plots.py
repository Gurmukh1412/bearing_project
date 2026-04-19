import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def run_evaluation(model, test_loader, inv_map, label_names, out_dir="."):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for xb, pb, yb in test_loader:
            xb, pb = xb.to(device), pb.to(device)
            logits = model(xb, pb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())
            
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Map back to original labels for reporting
    true_orig = np.vectorize(inv_map.get)(all_true)
    pred_orig = np.vectorize(inv_map.get)(all_preds)
    
    print("\nClassification Report:")
    print(classification_report(true_orig, pred_orig, target_names=[str(k) for k in sorted(inv_map.values())]))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(true_orig, pred_orig)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    print(f"Saved plot to {out_dir}/confusion_matrix.png")