"""
Distributed Image Classification with Federated Learning

This script is part of a BSc thesis evaluating the performance of federated learning
with regards to client size, non-iid vs iid data and client participation

Author: Anton Fredin and Marcus Johansson
Date: Spring 2026
Institution: KTH Royal Institute of Technology
Course: EF112X

For more information, see the corresponding thesis report:
""
TRITA: 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import time

from util import num_params
from data import fetch_dataset, iid_partition_loader, noniid_partition_loader
from torch.utils.data import random_split

np.random.seed(0)
torch.manual_seed(0)

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"| Using device: {device}")

# Hyperparameters
BATCH_SIZE = 50        
LR = 0.01             
LOCAL_EPOCHS = 3       
TARGET_ACCURACY = 0.90

if not os.path.exists('results'):
    os.makedirs('results')

print("Loading data...")
full_train_data, test_data = fetch_dataset()

# KRAV: Dela upp full_train_data i 90% träning och 20% validering
target_train_size = int(0.8 * len(full_train_data))
train_size = (target_train_size // 5000) * 5000 
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

# Nya dataloaders
central_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)


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

# Uppdaterad för att ta emot valfri dataloader (Train, Val eller Test)
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


def run_goal_1_baseline(epochs=5):
    """ 
    Returns the trained model, step/acc history, AND loss history for overfitting check
    """
    print(f"\n Goal 1: Centralized Baseline ({epochs} epochs)")
    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    acc_history = []
    step_history = []
    
    # NYA listor för loss
    train_loss_history = []
    val_loss_history = []
    
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
                acc = validate(model, val_loader)
                current_step = epoch + ((i + 1) / total_batches) 
                acc_history.append(acc)
                step_history.append(current_step)
                
                if acc >= TARGET_ACCURACY and not converged:
                    print(f"--> Converged (Acc >= {TARGET_ACCURACY}) at Epoch {current_step:.2f}")
                    converged = True
                model.train() 
                
        # NYTT: Räkna ut loss i slutet av varje epok för att kolla overfitting
        train_err = calculate_loss(model, central_train_loader)
        val_err = calculate_loss(model, val_loader)
        train_loss_history.append(train_err)
        val_loss_history.append(val_err)
                
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc_history[-1]:.4f} | Train Loss: {train_err:.4f} | Val Loss: {val_err:.4f}")
    
    print(f"Time Baseline: {time.time()-start:.1f}s")
    # Returnera loss-listorna också!
    return model, step_history, acc_history, train_loss_history, val_loss_history

def run_fl_experiment(name, K, loaders, rounds, C=1.0):
    print(f"\n Running: {name}")
    model = CNN().to(device)
    acc_hist = []
    
    num_selected = max(1, int(K * C))
    converged = False
    
    start = time.time()
    for r in range(rounds):
        if C < 1.0:
            indices = np.random.choice(range(K), num_selected, replace=False)
        else:
            indices = np.arange(K)
            
        model = fed_avg_round(model, loaders, indices, LR, LOCAL_EPOCHS)
        
        # Använd VAL_LOADER under träning
        acc = validate(model, val_loader)
        acc_hist.append(acc)
        
        if acc >= TARGET_ACCURACY and not converged:
            print(f"--> Converged at Round {r+1}")
            converged = True
            
        if (r+1) % 2 == 0: print(f"{name} | Round {r+1} | Val Acc: {acc:.4f}")
            
    print(f"Time {name}: {time.time()-start:.1f}s")
    # Returnerar även modellen för testning i slutet
    return model, acc_hist

if __name__ == "__main__":
    
    # Uppdaterat anrop som tar emot train_loss och val_loss (och kör 20 epoker)
    baseline_model, base_x, base_y, train_loss, val_loss = run_goal_1_baseline(epochs=20)
    
    # --- NY GRAF FÖR ÖVERINLÄRNING ---
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
    plt.show(block=False) # Visar grafen men låter programmet fortsätta till FL-experimenten
    # ---------------------------------
    
    # Goal 2: Scaling
    exp1_results = {}
    exp1_models = {}
    for K in [1, 5, 10, 20, 50, 100]:
        loaders = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=K)
        fl_model, fl_acc_hist = run_fl_experiment(f"K={K}", K, loaders, rounds=20)
        exp1_results[K] = fl_acc_hist
        exp1_models[K] = fl_model
        
    # Goal 3: Non-IID
    m_per_shard = int(len(train_data) / (10 * 2))
    non_iid_loaders = noniid_partition_loader(train_data, bsz=BATCH_SIZE, m_per_shard=m_per_shard, n_shards_per_client=2)
    non_iid_model, exp2_noniid = run_fl_experiment("Non-IID", 10, non_iid_loaders, rounds=20)
    
    # Goal 4: Participation (C)
    loaders_50 = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=50) # Korrigerat från 20 till 50
    exp3_results = {}
    exp3_models = {}
    for C in [0.02, 0.1, 0.2, 0.5, 1.0]:
        fl_model_c, fl_acc_hist_c = run_fl_experiment(f"C={C}", 50, loaders_50, rounds=20, C=C)
        exp3_results[C] = fl_acc_hist_c
        exp3_models[C] = fl_model_c

    print("\n--- Generating Plots ---")

    def save_plot(title, filename, base_x, base_y, experiments, labels_map=None):
        plt.figure(figsize=(10, 6))
        
        plt.plot(base_x, base_y, 'k--', linewidth=2, label='Baseline')
        
        for key, y_data in experiments.items():
            label = labels_map(key) if labels_map else str(key)
            x_axis = range(1, len(y_data) + 1)
            plt.plot(x_axis, y_data, label=label)
            
        plt.title(title)
        plt.xlabel("Effective Epochs / Rounds (Aligned)")
        plt.ylabel("Validation Accuracy") # Uppdaterat axelnamn
        plt.axhline(y=TARGET_ACCURACY, color='r', linestyle=':', label='Target 90%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/{filename}')
        plt.show()

    save_plot("Exp 1: Client Scaling", "exp1_scaling.png", base_x, base_y, exp1_results, lambda k: f'K={k}')
    
    exp2_data = {"IID (K=10)": exp1_results[10], "Non-IID (K=10)": exp2_noniid}
    save_plot("Exp 2: IID vs Non-IID", "exp2_distribution.png", base_x, base_y, exp2_data)
    
    save_plot("Exp 3: Participation (K=50)", "exp3_participation.png", base_x, base_y, exp3_results, lambda c: f'C={c}')
    
    # all plots combined
    plt.figure(figsize=(12, 8))
    plt.plot(base_x, base_y, 'k--', linewidth=2, label='Baseline')
    
    for k, v in exp1_results.items():
        plt.plot(range(1, len(v)+1), v, label=f'K={k}')
        
    plt.plot(range(1, len(exp2_noniid)+1), exp2_noniid, linestyle='--', linewidth=2, label='Non-IID')
    
    for c, v in exp3_results.items():
        plt.plot(range(1, len(v)+1), v, linestyle=':', label=f'C={c}')
        
    plt.title("All Experiments Combined")
    plt.xlabel("Effective Epochs / Rounds")
    plt.ylabel("Validation Accuracy") # Uppdaterat axelnamn
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/all_experiments_combined.png')
    plt.show()

    # ==========================================================
    # KRAV: SLUTGILTIG UTVÄRDERING PÅ TEST-DATAN (LÅST I VALVET)
    # ==========================================================
    print("\n" + "="*50)
    print(" OFFICIELLA RESULTAT PÅ ISOLERAD TEST-DATA")
    print("="*50)
    
    baseline_test_acc = validate(baseline_model, test_loader)
    print(f"Baseline (Centralized): \t{baseline_test_acc:.4f}")
    
    print("\n--- Goal 2: Client Scaling ---")
    for K, model in exp1_models.items():
        test_acc = validate(model, test_loader)
        print(f"K={K}: \t\t\t\t{test_acc:.4f}")
        
    print("\n--- Goal 3: IID vs Non-IID ---")
    print(f"Non-IID (K=10): \t\t{validate(non_iid_model, test_loader):.4f}")
    
    print("\n--- Goal 4: Participation (K=50) ---")
    for C, model in exp3_models.items():
        test_acc = validate(model, test_loader)
        print(f"C={C}: \t\t\t\t{test_acc:.4f}")