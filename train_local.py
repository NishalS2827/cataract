"""
Deep Multi-Layer Neural Network for Cataract Clinical Decision Support (CDSS)
Architecture: Input (54) -> Dense(64) + ReLU -> Dense(32) + ReLU -> Dense(1) + Sigmoid
Optimized with Adam Optimizer & L2 Regularization
Extracts multi-zone spatial pooling, local binary patterns, and chromatic sclerosis.
Runs entirely on Drive E: with zero external C: downloads.
"""

import os
import json
import random
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# 1. ADVANCED MULTIMODAL OPTICAL FEATURE EXTRACTOR (54 FEATURES)
# -------------------------------------------------------------------------
def extract_deep_optical_features(image_path):
    """
    Extracts 54 clinical ophthalmic features:
    - 4x4 spatial grid opacity density (16 features)
    - 4x4 spatial grid variance / cloudiness (16 features)
    - Local Binary Pattern (LBP) texture descriptors (8 features)
    - Multiscale gradient magnitude & Laplacian scattering (4 features)
    - Color chromaticity & nuclear sclerosis ratios (5 features)
    - Intensity quantiles (5 features)
    """
    img = Image.open(image_path).convert('RGB').resize((128, 128))
    arr_rgb = np.array(img, dtype=np.float32)
    
    # Isolate crystalline lens area (center 70% ROI)
    h, w, _ = arr_rgb.shape
    r_start, r_end = int(h * 0.15), int(h * 0.85)
    c_start, c_end = int(w * 0.15), int(w * 0.85)
    roi_rgb = arr_rgb[r_start:r_end, c_start:c_end]
    
    # Grayscale conversion
    roi_gray = (0.299 * roi_rgb[:, :, 0] + 0.587 * roi_rgb[:, :, 1] + 0.114 * roi_rgb[:, :, 2])
    
    features = []
    
    # A. 4x4 Spatial Grid Pooling (16 regions: center nuclear vs peripheral cortical)
    rh, rw = roi_gray.shape
    bh, bw = rh // 4, rw // 4
    grid_means, grid_vars = [], []
    for i in range(4):
        for j in range(4):
            block = roi_gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            grid_means.append(float(np.mean(block)) / 255.0)
            grid_vars.append(float(np.std(block)) / 128.0)
    features.extend(grid_means)  # 16
    features.extend(grid_vars)   # 16
    
    # B. Local Binary Pattern (LBP) Micro-texture Approximation (8 bins)
    padded = np.pad(roi_gray, 1, mode='edge')
    center = padded[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dr, dc) in enumerate(shifts):
        neighbor = padded[1+dr:1+dr+center.shape[0], 1+dc:1+dc+center.shape[1]]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 255), density=True)
    features.extend(lbp_hist.tolist())  # 8
    
    # C. Multiscale Gradients (Cortical Spokes & Edge Scatter) (4 features)
    gx = np.abs(np.roll(roi_gray, 1, axis=1) - np.roll(roi_gray, -1, axis=1))
    gy = np.abs(np.roll(roi_gray, 1, axis=0) - np.roll(roi_gray, -1, axis=0))
    grad_mag = np.sqrt(gx**2 + gy**2)
    features.append(float(np.mean(grad_mag)) / 128.0)
    features.append(float(np.std(grad_mag)) / 64.0)
    
    lap = (-4 * roi_gray + np.roll(roi_gray, 1, axis=0) + np.roll(roi_gray, -1, axis=0)
           + np.roll(roi_gray, 1, axis=1) + np.roll(roi_gray, -1, axis=1))
    features.append(float(np.mean(np.abs(lap))) / 128.0)
    features.append(float(np.var(lap)) / 10000.0)
    
    # D. Color Chromaticity & Sclerosis Shifts (5 features)
    r_mean = float(np.mean(roi_rgb[:, :, 0])) + 1e-5
    g_mean = float(np.mean(roi_rgb[:, :, 1])) + 1e-5
    b_mean = float(np.mean(roi_rgb[:, :, 2])) + 1e-5
    tot = r_mean + g_mean + b_mean
    features.append(r_mean / tot)  # Normalized Red
    features.append(g_mean / tot)  # Normalized Green
    features.append(b_mean / tot)  # Normalized Blue
    features.append(r_mean / g_mean)  # Yellow shift
    features.append(r_mean / b_mean)  # Amber shift
    
    # E. Intensity Quantiles (5 features: 10th, 25th, 50th, 75th, 90th percentile)
    q = np.quantile(roi_gray, [0.10, 0.25, 0.50, 0.75, 0.90]) / 255.0
    features.extend(q.tolist())
    
    return np.array(features, dtype=np.float32)

# -------------------------------------------------------------------------
# 2. DATASET LOADER
# -------------------------------------------------------------------------
def load_dataset_partition(base_dir, split_name):
    split_dir = os.path.join(base_dir, split_name)
    X, y = [], []
    
    cataract_dir = os.path.join(split_dir, 'cataract')
    if os.path.exists(cataract_dir):
        for f in os.listdir(cataract_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                feat = extract_deep_optical_features(os.path.join(cataract_dir, f))
                X.append(feat)
                y.append(1.0)
                
    normal_dir = os.path.join(split_dir, 'normal')
    if os.path.exists(normal_dir):
        for f in os.listdir(normal_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                feat = extract_deep_optical_features(os.path.join(normal_dir, f))
                X.append(feat)
                y.append(0.0)
                
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# -------------------------------------------------------------------------
# 3. DEEP MULTI-LAYER NEURAL NETWORK (ADAM OPTIMIZER)
# -------------------------------------------------------------------------
class DeepCataractNeuralNet:
    """
    3-Layer Deep Neural Architecture:
    Input (D=54) -> Hidden 1 (H1=64, ReLU) -> Hidden 2 (H2=32, ReLU) -> Output (1, Sigmoid)
    """
    def __init__(self, input_dim=54, h1_dim=64, h2_dim=32):
        np.random.seed(42)
        # He (Kaiming) Weight Initialization
        self.W1 = np.random.randn(input_dim, h1_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, h1_dim), dtype=np.float32)
        
        self.W2 = np.random.randn(h1_dim, h2_dim) * np.sqrt(2.0 / h1_dim)
        self.b2 = np.zeros((1, h2_dim), dtype=np.float32)
        
        self.W3 = np.random.randn(h2_dim, 1) * np.sqrt(2.0 / h2_dim)
        self.b3 = np.zeros((1, 1), dtype=np.float32)
        
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.std = np.ones(input_dim, dtype=np.float32)
        
        # Adam Optimizer Moments
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3); self.vW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3); self.vb3 = np.zeros_like(self.b3)
        self.t = 0
        
    def fit_normalizer(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-7
        
    def normalize(self, X):
        return (X - self.mean) / self.std
        
    def relu(self, Z):
        return np.maximum(0, Z)
        
    def sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -15.0, 15.0)))
        
    def forward(self, X_norm):
        # Layer 1
        Z1 = np.dot(X_norm, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        # Layer 2
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.relu(Z2)
        
        # Layer 3 (Output)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.sigmoid(Z3)
        
        cache = (X_norm, Z1, A1, Z2, A2, Z3, A3)
        return A3.flatten(), cache

    def train(self, X_train, y_train, X_val, y_val, epochs=50, lr=0.003, l2_reg=1e-3, batch_size=32):
        self.fit_normalizer(X_train)
        X_train_n = self.normalize(X_train)
        X_val_n = self.normalize(X_val)
        y_train_2d = y_train.reshape(-1, 1)
        
        n_samples = len(X_train)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        print("=" * 68)
        print("     TRAINING DEEP MULTI-LAYER NEURAL NETWORK FOR CATARACT CDSS")
        print("=" * 68)
        print(f"[*] Architecture       : Input (54) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)")
        print(f"[*] Optimizer          : Adam (lr={lr}, beta1={beta1}, beta2={beta2})")
        print(f"[*] Regularization     : L2 Weight Decay ({l2_reg})")
        print(f"[*] Training Samples   : {n_samples} (Balanced 1:1)")
        print(f"[*] Validation Samples : {len(X_val)}")
        print(f"[*] Total Epochs       : {epochs} | Batch Size: {batch_size}")
        print("-" * 68)
        print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10}")
        print("-" * 68)
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            # Mini-batch shuffling
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                
                xb = X_train_n[batch_idx]
                yb = y_train_2d[batch_idx]
                m = len(xb)
                
                # Forward Pass
                _, (X_norm, Z1, A1, Z2, A2, Z3, A3) = self.forward(xb)
                
                # Backpropagation
                dZ3 = A3 - yb  # (m, 1)
                dW3 = (1.0 / m) * np.dot(A2.T, dZ3) + l2_reg * self.W3
                db3 = (1.0 / m) * np.sum(dZ3, axis=0, keepdims=True)
                
                dA2 = np.dot(dZ3, self.W3.T)
                dZ2 = dA2 * (Z2 > 0)
                dW2 = (1.0 / m) * np.dot(A1.T, dZ2) + l2_reg * self.W2
                db2 = (1.0 / m) * np.sum(dZ2, axis=0, keepdims=True)
                
                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * (Z1 > 0)
                dW1 = (1.0 / m) * np.dot(X_norm.T, dZ1) + l2_reg * self.W1
                db1 = (1.0 / m) * np.sum(dZ1, axis=0, keepdims=True)
                
                # Adam Updates
                self.t += 1
                for param, grad, m_arr, v_arr in [
                    (self.W1, dW1, self.mW1, self.vW1), (self.b1, db1, self.mb1, self.vb1),
                    (self.W2, dW2, self.mW2, self.vW2), (self.b2, db2, self.mb2, self.vb2),
                    (self.W3, dW3, self.mW3, self.vW3), (self.b3, db3, self.mb3, self.vb3)
                ]:
                    m_arr[:] = beta1 * m_arr + (1.0 - beta1) * grad
                    v_arr[:] = beta2 * v_arr + (1.0 - beta2) * (grad ** 2)
                    m_hat = m_arr / (1.0 - beta1 ** self.t)
                    v_hat = v_arr / (1.0 - beta2 ** self.t)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    
            # Epoch Metrics
            train_preds, _ = self.forward(X_train_n)
            train_loss = -np.mean(y_train * np.log(train_preds + 1e-7) + (1.0 - y_train) * np.log(1.0 - train_preds + 1e-7))
            train_acc = np.mean((train_preds >= 0.5) == y_train) * 100.0
            
            val_preds, _ = self.forward(X_val_n)
            val_loss = -np.mean(y_val * np.log(val_preds + 1e-7) + (1.0 - y_val) * np.log(1.0 - val_preds + 1e-7))
            val_acc = np.mean((val_preds >= 0.5) == y_val) * 100.0
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(f"{epoch:^8} | {train_loss:^12.4f} | {train_acc:^9.1f}% | {val_loss:^10.4f} | {val_acc:^9.1f}%")
                
        print("-" * 68)
        print(f"[OK] Training Completed! Peak Validation Accuracy: {best_val_acc:.2f}%")

    def evaluate(self, X_test, y_test):
        X_test_n = self.normalize(X_test)
        probs, _ = self.forward(X_test_n)
        preds = (probs >= 0.5).astype(np.float32)
        
        tp = int(np.sum((preds == 1) & (y_test == 1)))
        tn = int(np.sum((preds == 0) & (y_test == 0)))
        fp = int(np.sum((preds == 1) & (y_test == 0)))
        fn = int(np.sum((preds == 0) & (y_test == 1)))
        
        total = len(y_test)
        accuracy = (tp + tn) / max(total, 1) * 100.0
        sensitivity = tp / max(tp + fn, 1) * 100.0
        specificity = tn / max(tn + fp, 1) * 100.0
        precision = tp / max(tp + fp, 1) * 100.0
        f1 = (2 * precision * sensitivity) / max(precision + sensitivity, 1e-5)
        
        print("\n" + "=" * 68)
        print("          INDEPENDENT TEST SET CLINICAL EVALUATION (UNSEEN EYES)")
        print("=" * 68)
        print(f"Total Test Cases   : {total} (Cataract: {tp+fn}, Normal: {tn+fp})")
        print(f"Accuracy           : {accuracy:.2f}%")
        print(f"Sensitivity (Recall): {sensitivity:.2f}% (Cataract True Positive Detection)")
        print(f"Specificity        : {specificity:.2f}% (Normal True Negative Rate)")
        print(f"Precision          : {precision:.2f}%")
        print(f"F1-Score           : {f1:.2f}%")
        print("-" * 68)
        print("Confusion Matrix:")
        print(f"                  Predicted Normal    Predicted Cataract")
        print(f"  Actual Normal   :      {tn:<10}          {fp:<10}")
        print(f"  Actual Cataract :      {fn:<10}          {tp:<10}")
        print("=" * 68)
        return accuracy

    def save(self, filepath="cataract_trained_model.json"):
        model_data = {
            "model_type": "Deep Multi-Layer Neural Network (3-Layer)",
            "architecture": [54, 64, 32, 1],
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"[OK] Saved Deep Neural Model to: {filepath}")

# -------------------------------------------------------------------------
# 4. MANUAL INTERACTIVE CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual Deep Cataract Neural Network Training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default 0.003)")
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr

    print("=" * 68)
    print("   AI CATARACT CDSS — MANUAL DEEP NEURAL NETWORK TRAINING")
    print("=" * 68)

    if epochs is None:
        try:
            val_in = input("Enter number of Epochs [Press Enter for 60]: ").strip()
            epochs = int(val_in) if val_in else 60
        except (ValueError, EOFError):
            epochs = 60

    if lr is None:
        try:
            val_in = input("Enter Learning Rate [Press Enter for 0.003]: ").strip()
            lr = float(val_in) if val_in else 0.003
        except (ValueError, EOFError):
            lr = 0.003

    print(f"\n[*] Selected Parameters: Epochs = {epochs}, Learning Rate = {lr}")
    dataset_base = os.path.join("data", "dataset")

    print("[*] Extracting 54-D deep optical features from Training Set...")
    X_train, y_train = load_dataset_partition(dataset_base, "train")

    print("[*] Extracting deep optical features from Validation Set...")
    X_val, y_val = load_dataset_partition(dataset_base, "val")

    print("[*] Extracting deep optical features from Test Set...")
    X_test, y_test = load_dataset_partition(dataset_base, "test")

    net = DeepCataractNeuralNet(input_dim=X_train.shape[1], h1_dim=64, h2_dim=32)
    net.train(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)
    
    test_acc = net.evaluate(X_test, y_test)
    net.save("cataract_trained_model.json")
    print("\n[OK] Deep Model Manual Training Completed Successfully!")
