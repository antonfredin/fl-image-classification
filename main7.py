"""
Distributed Image Classification with Federated Learning

This script is part of a BSc thesis evaluating the performance of federated learning
with regards to client size, non-iid vs iid data and client participation.

Author: Anton Fredin and Marcus Johansson
Date: Spring 2026
Institution: KTH Royal Institute of Technology
Course: EF112X

For more information, see the corresponding thesis report.
TRITA: 
"""

import os
import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from util import num_params
from data import fetch_dataset, iid_partition_loader, noniid_partition_loader

# ---------------------------------------------------------------------------
# Configuration and Hyperparameters
# ---------------------------------------------------------------------------
np.random.seed(0)
torch.manual_seed(0)

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"| Using device: {device}")

BATCH_SIZE = 50        
LR = 0.01             
LOCAL_EPOCHS = 3       
TARGET_ACCURACY = 0.90
EXP3_BATCH_SIZE = 10

if not os.path.exists('results'):
    os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------------------------
# Data Loading and Partitioning
# ---------------------------------------------------------------------------
print("Loading data...")
full_train_data, test_data = fetch_dataset()

# Ensure train_size is divisible by (600 clients * 50 batch size) = 30000
target_train_size = int(0.8 * len(full_train_data))
train_size = (target_train_size // 30000) * 30000 
if train_size == 0: 
    train_size = 30000 
val_size = len(full_train_data) - train_size

train_data, val_data = random_split(full_train_data, [train_size, val_size])

central_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5) 
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(p=0.5) 
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Shared Model Initialization
# ---------------------------------------------------------------------------
# To ensure a scientifically fair comparison across all experiments,
# we generate and store a master set of initialized weights.
initial_model_instance = CNN()
INITIAL_MODEL_STATE = copy.deepcopy(initial_model_instance.state_dict())

# ---------------------------------------------------------------------------
# Training and Evaluation Utilities
# ---------------------------------------------------------------------------
def validate(model, dataloader):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.shape[0]
    return correct / total

def calculate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def train_client(client_loader, global_model, num_local_epochs, lr):
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    
    for _ in range(num_local_epochs):
        for x, y in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(x), y)
            loss.backward()
            optimizer.step()
    return local_model.state_dict()

def fed_avg_round(global_model, client_loaders, client_indices, lr, local_epochs):
    global_weights = global_model.state_dict()
    local_weights = []
    
    for idx in client_indices:
        w = train_client(client_loaders[idx], global_model, local_epochs, lr)
        local_weights.append(w)
    
    avg_weights = copy.deepcopy(global_weights)
    for key in avg_weights.keys():
        avg_weights[key] = torch.stack([w[key] for w in local_weights]).mean(0)
        
    global_model.load_state_dict(avg_weights)
    return global_model

# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------
def run_goal_1_baseline(epochs=5):
    print(f"\n Goal 1: Centralized Baseline ({epochs} epochs)")
    model = CNN().to(device)
    model.load_state_dict(INITIAL_MODEL_STATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    val_acc_history, test_acc_history, step_history = [], [], []
    train_loss_history, val_loss_history = [], []
    converged = False
    
    LOGS_PER_EPOCH = 10
    total_batches = len(central_train_loader)
    log_interval = max(1, total_batches // LOGS_PER_EPOCH)
    
    start = time.time()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(central_train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % log_interval == 0:
                val_acc = validate(model, val_loader)
                test_acc = validate(model, test_loader)
                
                current_step = epoch + ((i + 1) / total_batches) 
                val_acc_history.append(val_acc)
                test_acc_history.append(test_acc)
                step_history.append(current_step)
                
                if test_acc >= TARGET_ACCURACY and not converged:
                    print(f"--> Converged (Test Acc >= {TARGET_ACCURACY}) at Epoch {current_step:.2f}")
                    converged = True
                model.train() 
                
        train_err = calculate_loss(model, central_train_loader)
        val_err = calculate_loss(model, val_loader)
        train_loss_history.append(train_err)
        val_loss_history.append(val_err)
                
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc_history[-1]:.4f} | Test Acc: {test_acc_history[-1]:.4f}")
    
    print(f"Time Baseline: {time.time()-start:.1f}s")
    return model, step_history, val_acc_history, test_acc_history, train_loss_history, val_loss_history

def run_fl_experiment(name, K, loaders, rounds, C=1.0):
    print(f"\n Running: {name}")
    model = CNN().to(device)
    model.load_state_dict(INITIAL_MODEL_STATE)
    
    val_acc_hist = []
    test_acc_hist = []
    
    num_selected = max(1, int(K * C))
    converged = False
    
    start = time.time()
    for r in range(rounds):
        if C < 1.0:
            indices = np.random.choice(range(K), num_selected, replace=False)
        else:
            indices = np.arange(K)
            
        model = fed_avg_round(model, loaders, indices, LR, LOCAL_EPOCHS)
        
        val_acc = validate(model, val_loader)
        test_acc = validate(model, test_loader)
        
        val_acc_hist.append(val_acc)
        test_acc_hist.append(test_acc)
        
        if test_acc >= TARGET_ACCURACY and not converged:
            print(f"--> Converged at Round {r+1}")
            converged = True
            
        if (r+1) % 2 == 0: 
            print(f"{name} | Round {r+1} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            
    print(f"Time {name}: {time.time()-start:.1f}s")
    return model, val_acc_hist, test_acc_hist

# ---------------------------------------------------------------------------
# Plotting Utility
# ---------------------------------------------------------------------------
def save_plot(title, filename, base_x, base_val, base_test, exp_val, exp_test, labels_map=None):
    plt.figure(figsize=(12, 7))
    
    # Plot Baseline
    plt.plot(base_x, base_test, '-', linewidth=2, label='Baseline (Test)', color='black')
    plt.plot(base_x, base_val, '--', linewidth=2, label='Baseline (Val)', color='black', alpha=0.7)
    
    # Plot FL Experiments
    for key in exp_test.keys():
        label = labels_map(key) if labels_map else str(key)
        x_axis = range(1, len(exp_test[key]) + 1)
        
        p = plt.plot(x_axis, exp_test[key], '-', linewidth=1.5, label=f"{label} (Test)")
        color = p[0].get_color()
        plt.plot(x_axis, exp_val[key], '--', color=color, linewidth=1.5, label=f"{label} (Val)", alpha=0.7)
        
    plt.title(title)
    plt.xlabel("Communication rounds")
    plt.ylabel("Accuracy")
    plt.axhline(y=TARGET_ACCURACY, color='r', linestyle=':', label='Target 90%')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{filename}')
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Run Baseline (100 Epochs)
    baseline_model, base_x, base_val, base_test, train_loss, val_loss = run_goal_1_baseline(epochs=100)
    
    # Overfitting Check Plot
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(train_loss) + 1)
    plt.plot(epochs_range, train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    plt.title("Baseline Overfitting Check (Loss)")
    plt.xlabel("Epochs")
    plt.ylabel("Error (Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/baseline_overfitting.png')
    plt.close()
    
    # Exp 1: Client Scaling (20 Rounds)
    exp1_val = {}
    exp1_test = {}
    for K in [1, 5, 10, 20, 50, 100]:
        loaders = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=K)
        fl_model, val_h, test_h = run_fl_experiment(f"K={K}", K, loaders, rounds=20)
        exp1_val[K] = val_h
        exp1_test[K] = test_h
        
    # Exp 2: Non-IID Distribution (20 Rounds)
    m_per_shard = int(len(train_data) / (10 * 2))
    non_iid_loaders = noniid_partition_loader(train_data, bsz=BATCH_SIZE, m_per_shard=m_per_shard, n_shards_per_client=2)
    non_iid_model, exp2_noniid_val, exp2_noniid_test = run_fl_experiment("Non-IID", 10, non_iid_loaders, rounds=20)
    
    # Exp 3: Participation Scaling (100 Rounds)
    loaders_600 = iid_partition_loader(train_data, bsz=EXP3_BATCH_SIZE, n_clients=600)
    exp3_val = {}
    exp3_test = {}
    exp3_models = {}
    for C in [0.002, 0.02, 0.1, 0.2, 0.5, 1.0]:
        fl_model_c, val_h_c, test_h_c = run_fl_experiment(f"C={C}", 600, loaders_600, rounds=100, C=C)
        exp3_val[C] = val_h_c
        exp3_test[C] = test_h_c
        exp3_models[C] = fl_model_c

    # ---------------------------------------------------------------------------
    # Plot Generation & Data Export
    # ---------------------------------------------------------------------------
    print("\n--- Generating Plots ---")

    # Slice baseline up to 20 for Exp 1 & 2 to prevent X-axis stretching.
    # LOGS_PER_EPOCH is 10, so 20 epochs = 200 logged steps. 
    base_slice_idx = 200 
    sliced_base_x = base_x[:base_slice_idx]
    sliced_base_val = base_val[:base_slice_idx]
    sliced_base_test = base_test[:base_slice_idx]

    save_plot("Exp 1: Client Scaling", "exp1_scaling.png", 
              sliced_base_x, sliced_base_val, sliced_base_test, exp1_val, exp1_test, lambda k: f'K={k}')
    
    exp2_v_data = {"IID (K=10)": exp1_val[10], "Non-IID (K=10)": exp2_noniid_val}
    exp2_t_data = {"IID (K=10)": exp1_test[10], "Non-IID (K=10)": exp2_noniid_test}
    save_plot("Exp 2: IID vs Non-IID", "exp2_distribution.png", 
              sliced_base_x, sliced_base_val, sliced_base_test, exp2_v_data, exp2_t_data)
    
    # Use full baseline (100 epochs) for Exp 3
    save_plot("Exp 3: Participation (K=600)", "exp3_participation.png", 
              base_x, base_val, base_test, exp3_val, exp3_test, lambda c: f'C={c}')

    # Output Official Test Results
    print("\n" + "="*50)
    print(" OFFICIAL TEST SET RESULTS")
    print("="*50)
    
    baseline_test_acc = validate(baseline_model, test_loader)
    print(f"Baseline (Centralized): \t{baseline_test_acc:.4f}")
    
    print("\n--- Goal 4: Participation (K=600) ---")
    for C, model in exp3_models.items():
        test_acc = validate(model, test_loader)
        print(f"C={C}: \t\t\t\t{test_acc:.4f}")

    # Save numeric outputs to text
    with open("results/final_test_accuracy.txt", "w") as f:
        f.write("OFFICIAL TEST SET RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Baseline (Centralized): \t{baseline_test_acc:.4f}\n\n")
        f.write("--- Goal 4: Participation (K=600) ---\n")
        for C, model in exp3_models.items():
            test_acc = validate(model, test_loader)
            f.write(f"C={C}: \t\t\t\t{test_acc:.4f}\n")
            
    # Save raw arrays and dictionaries to a pickle file for future replotting
    raw_data_export = {
        "baseline": {"x": base_x, "val": base_val, "test": base_test, "train_loss": train_loss, "val_loss": val_loss},
        "exp1": {"val": exp1_val, "test": exp1_test},
        "exp2": {"val": exp2_v_data, "test": exp2_t_data},
        "exp3": {"val": exp3_val, "test": exp3_test}
    }
    
    with open("results/experiment_raw_data.pkl", "wb") as f:
        pickle.dump(raw_data_export, f)

    print("\n[INFO] Results saved to 'results/final_test_accuracy.txt'")
    print("[INFO] Raw plotting data successfully serialized to 'results/experiment_raw_data.pkl'")