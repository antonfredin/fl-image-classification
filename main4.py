import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import time

# Importer från dina filer
from util import num_params
from data import fetch_dataset, iid_partition_loader, noniid_partition_loader

# =============================================================================
# 1. Konfiguration & Setup
# =============================================================================
np.random.seed(0)
torch.manual_seed(0)

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"| Using device: {device}")

# --- HYPERPARAMETRAR ---
BATCH_SIZE = 50        # Snabbare på CPU (färre uppdateringar/runda)
LR = 0.01              # Anpassad learning rate
LOCAL_EPOCHS = 1       # 1 Runda FL = 1 Epok Baseline
TARGET_ACCURACY = 0.90 # Enligt arbetsplan

if not os.path.exists('results'):
    os.makedirs('results')

print("Loading data...")
train_data, test_data = fetch_dataset()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
central_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# =============================================================================
# 2. Modell: SimpleCNN MED DROPOUT
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5) 
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(128, 32)
        
        # KRAV: Dropout lager
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

# =============================================================================
# 3. Hjälpfunktioner
# =============================================================================
def validate(model):
    """ Returnerar accuracy (0.0 - 1.0) """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.shape[0]
    return correct / total

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

# =============================================================================
# 4. Experiment
# =============================================================================

def run_goal_1_baseline(epochs=5):
    """ 
    Goal 1: Baseline 
    Returnerar TVÅ listor: (Tidsteg, Accuracy) för att kunna plotta med hög upplösning.
    """
    print(f"\n--- Goal 1: Centralized Baseline ({epochs} epochs) ---")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    acc_history = []
    step_history = [] # Sparar t.ex. 0.1, 0.2, 0.3...
    converged = False
    
    # High Resolution: Logga 10 gånger per epok
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
            
            # Logga del-resultat
            if (i + 1) % log_interval == 0:
                acc = validate(model)
                
                # Räkna ut exakt var vi är (t.ex. Epok 1.5)
                current_step = epoch + ((i + 1) / total_batches)
                
                acc_history.append(acc)
                step_history.append(current_step)
                
                if acc >= TARGET_ACCURACY and not converged:
                    print(f"--> Converged (Acc >= {TARGET_ACCURACY}) at Epoch {current_step:.2f}")
                    converged = True
                model.train() 
                
        print(f"Epoch {epoch+1}/{epochs} | Acc: {acc_history[-1]:.4f}")
    
    print(f"Time Baseline: {time.time()-start:.1f}s")
    return step_history, acc_history

def run_fl_experiment(name, K, loaders, rounds, C=1.0):
    print(f"\n--- Running: {name} ---")
    model = SimpleCNN().to(device)
    acc_hist = []
    
    num_selected = max(1, int(K * C))
    converged = False
    
    start = time.time()
    for r in range(rounds):
        # Välj klienter
        if C < 1.0:
            indices = np.random.choice(range(K), num_selected, replace=False)
        else:
            indices = np.arange(K)
            
        model = fed_avg_round(model, loaders, indices, LR, LOCAL_EPOCHS)
        acc = validate(model)
        acc_hist.append(acc)
        
        if acc >= TARGET_ACCURACY and not converged:
            print(f"--> Converged at Round {r+1}")
            converged = True
            
        if (r+1) % 2 == 0: print(f"{name} | Round {r+1} | Acc: {acc:.4f}")
            
    print(f"Time {name}: {time.time()-start:.1f}s")
    return acc_hist

# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    
    # 1. Baseline (High Res)
    # Vi får tillbaka både X-axel (steg) och Y-axel (acc)
    base_x, base_y = run_goal_1_baseline(epochs=5)
    
    # 2. FL Experiment (Standard Res: 1 punkt per runda)
    # Goal 2: Scaling
    exp1_results = {}
    for K in [5, 10, 20, 50]:
        loaders = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=K)
        exp1_results[K] = run_fl_experiment(f"K={K}", K, loaders, rounds=10)
        
    # Goal 3: Non-IID
    m_per_shard = int(len(train_data) / (10 * 2))
    non_iid_loaders = noniid_partition_loader(train_data, bsz=BATCH_SIZE, m_per_shard=m_per_shard, n_shards_per_client=2)
    exp2_noniid = run_fl_experiment("Non-IID", 10, non_iid_loaders, rounds=10)
    
    # Goal 4: Participation (C)
    loaders_50 = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=50)
    exp3_results = {}
    for C in [0.1, 0.5, 1.0]:
        exp3_results[C] = run_fl_experiment(f"C={C}", 50, loaders_50, rounds=10, C=C)

    # --- PLOTTING ---
    print("\n--- Generating Plots ---")

    def save_plot(title, filename, base_x, base_y, experiments, labels_map=None):
        plt.figure(figsize=(10, 6))
        
        # Plotta Baseline med exakta tidssteg (Mjuk kurva!)
        plt.plot(base_x, base_y, 'k--', linewidth=2, label='Baseline')
        
        # Plotta Experiment (Heltal: 1, 2, 3...)
        for key, y_data in experiments.items():
            label = labels_map(key) if labels_map else str(key)
            # Skapa X-axel för FL: [1, 2, 3...]
            x_axis = range(1, len(y_data) + 1)
            plt.plot(x_axis, y_data, label=label)
            
        plt.title(title)
        plt.xlabel("Effective Epochs / Rounds (Aligned)")
        plt.ylabel("Accuracy")
        plt.axhline(y=TARGET_ACCURACY, color='r', linestyle=':', label='Target 90%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/{filename}')
        plt.show()

    # Plot 1: Scaling
    save_plot("Exp 1: Client Scaling", "exp1_scaling.png", base_x, base_y, exp1_results, lambda k: f'K={k}')
    
    # Plot 2: Non-IID
    exp2_data = {"IID (K=10)": exp1_results[10], "Non-IID (K=10)": exp2_noniid}
    save_plot("Exp 2: IID vs Non-IID", "exp2_distribution.png", base_x, base_y, exp2_data)
    
    # Plot 3: Participation
    save_plot("Exp 3: Participation (K=50)", "exp3_participation.png", base_x, base_y, exp3_results, lambda c: f'C={c}')
    
    # Plot 4: Combined (ALLA i samma)
    plt.figure(figsize=(12, 8))
    plt.plot(base_x, base_y, 'k--', linewidth=2, label='Baseline')
    
    for k, v in exp1_results.items():
        plt.plot(range(1, len(v)+1), v, label=f'K={k}')
        
    plt.plot(range(1, len(exp2_noniid)+1), exp2_noniid, linestyle='--', linewidth=2, label='Non-IID')
    
    for c, v in exp3_results.items():
        plt.plot(range(1, len(v)+1), v, linestyle=':', label=f'C={c}')
        
    plt.title("All Experiments Combined")
    plt.xlabel("Effective Epochs / Rounds")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/all_experiments_combined.png')
    plt.show()

    print("Körning klar. Alla grafer sparade.")