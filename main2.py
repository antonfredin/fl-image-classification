import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Imports from your separate files
from util import num_params
from data import fetch_dataset, iid_partition_loader, noniid_partition_loader

# =============================================================================
# Configuration & Setup
# =============================================================================
np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"| Using device: {device}")

# Hyperparameters
BATCH_SIZE = 10
LR = 0.01
LOCAL_EPOCHS = 5  # "local stochastic gradient descent on each client" [cite: 13]
TARGET_ACCURACY = 0.90 # Goal 1 & 2 Test [cite: 38, 44]

# Load Data (MNIST) [cite: 21, 66]
train_data, test_data = fetch_dataset()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
central_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# =============================================================================
# Model Definition (CNN) [cite: 21]
# =============================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x

criterion = nn.CrossEntropyLoss()

# =============================================================================
# Core FL Functions (FedAvg) [cite: 13]
# =============================================================================
def validate(model):
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
    # "communicating the resulting model updates to a central server" [cite: 13]
    global_weights = global_model.state_dict()
    local_weights = []
    
    # Train selected clients
    for idx in client_indices:
        w = train_client(client_loaders[idx], global_model, local_epochs, lr)
        local_weights.append(w)
    
    # Aggregation (FedAvg)
    avg_weights = copy.deepcopy(global_weights)
    for key in avg_weights.keys():
        # Average weights: sum(client_weight) / num_participating
        avg_weights[key] = torch.stack([w[key] for w in local_weights]).mean(0)
        
    global_model.load_state_dict(avg_weights)
    return global_model

# =============================================================================
# Experiments as defined in Work Plan
# =============================================================================

def run_goal_1_baseline(epochs=20):
    """
    Goal 1: Baseline: Centralized Learning [cite: 30]
    Requirements: Reach 90% acc, report learning curves[cite: 33, 34, 38].
    """
    print(f"\n--- Goal 1: Centralized Baseline ---")
    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    acc_history = []
    for epoch in range(epochs):
        model.train()
        for x, y in central_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        acc = validate(model)
        acc_history.append(acc)
        print(f"Epoch {epoch+1} | Acc: {acc:.4f}")
        if acc >= TARGET_ACCURACY:
            print(f"Goal 1 Reached: >90% accuracy at epoch {epoch+1}")
            
    return acc_history

def run_goal_2_client_scaling(client_counts=[5, 10, 20, 50], rounds=50):
    """
    Goal 2: Experiment 1: Effect of Number of Clients (K) [cite: 40]
    "Evaluate how model performance scales as K increases" 
    "Robustness: Find max clients before FedAvg diverges" [cite: 46]
    """
    print(f"\n--- Goal 2: Client Scaling (IID) ---")
    results = {}
    
    for K in client_counts:
        print(f"Running for K={K} clients...")
        # "total dataset size remains fixed" -> Partition into K parts 
        loaders = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=K)
        
        model = CNN().to(device)
        acc_hist = []
        
        for r in range(rounds):
            # For Exp 1, we often look at full participation or fixed fraction.
            # Assuming full participation to test "Robustness" of K purely.
            selected_indices = np.arange(K) 
            
            model = fed_avg_round(model, loaders, selected_indices, LR, LOCAL_EPOCHS)
            acc = validate(model)
            acc_hist.append(acc)
            
            if r % 5 == 0:
                print(f"K={K} | Round {r} | Acc: {acc:.4f}")
        
        results[K] = acc_hist
        
    return results

def run_goal_3_data_distribution(K=10, rounds=50):
    """
    Goal 3: Experiment 2: Effect of Data Distribution [cite: 49]
    Compare IID vs Non-IID[cite: 50, 51, 52].
    """
    print(f"\n--- Goal 3: Data Distribution (Non-IID vs IID) ---")
    
    # 1. Non-IID Setup [cite: 52]
    # "partitioned so that each client only has specific digits"
    # m_per_shard calculation: Total 60000 images. K clients. 
    # To use existing loader logic, we need to adjust parameters carefully.
    # If K=10, each client needs ~6000 images.
    m_per_shard = int(len(train_data) / (K * 2)) # 2 shards per client
    non_iid_loaders = noniid_partition_loader(train_data, bsz=BATCH_SIZE, 
                                              m_per_shard=m_per_shard, 
                                              n_shards_per_client=2)
    
    # 2. IID Setup (Control) [cite: 51]
    iid_loaders = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=K)
    
    experiments = {
        "Non-IID": non_iid_loaders,
        "IID": iid_loaders
    }
    
    results = {}
    
    for label, loaders in experiments.items():
        print(f"Running {label} distribution...")
        model = CNN().to(device)
        acc_hist = []
        for r in range(rounds):
            # Use full participation or C=1.0 for fair comparison of distribution effect
            selected_indices = np.arange(K)
            model = fed_avg_round(model, loaders, selected_indices, LR, LOCAL_EPOCHS)
            acc = validate(model)
            acc_hist.append(acc)
            if r % 5 == 0: print(f"{label} | Round {r} | Acc: {acc:.4f}")
        results[label] = acc_hist
        
    return results

def run_goal_4_partial_participation(K=100, fractions=[0.1, 0.5], rounds=50):
    """
    Goal 4: Experiment 3: Effect of Partial Client Participation (Optional) [cite: 57]
    "Varying the client fraction C" [cite: 58]
    """
    print(f"\n--- Goal 4: Partial Participation (C) ---")
    loaders = iid_partition_loader(train_data, bsz=BATCH_SIZE, n_clients=K)
    results = {}
    
    for C in fractions:
        num_selected = int(K * C)
        print(f"Running C={C} ({num_selected} clients/round)...")
        model = CNN().to(device)
        acc_hist = []
        
        for r in range(rounds):
            # "Only a random fraction... is selected" [cite: 58]
            selected_indices = np.random.choice(range(K), num_selected, replace=False)
            model = fed_avg_round(model, loaders, selected_indices, LR, LOCAL_EPOCHS)
            acc = validate(model)
            acc_hist.append(acc)
            if r % 5 == 0: print(f"C={C} | Round {r} | Acc: {acc:.4f}")
            
        results[f"C={C}"] = acc_hist
        
    return results

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    
    # 1. Run Baseline (Goal 1)
    baseline_acc = run_goal_1_baseline(epochs=20)
    
    # 2. Run Client Scaling Experiment (Goal 2)
    # Using client counts from Risk Analysis/Tests [cite: 48]
    exp1_results = run_goal_2_client_scaling(client_counts=[5, 10, 20, 50], rounds=30)
    
    # 3. Run Data Distribution Experiment (Goal 3)
    exp2_results = run_goal_3_data_distribution(K=10, rounds=30)
    
    # 4. (Optional) Partial Participation (Goal 4) [cite: 60]
    # Uncomment to run if time permits
    # exp3_results = run_goal_4_partial_participation(K=100, fractions=[0.1, 0.2], rounds=30)

    # Plotting for Work Plan Reporting [cite: 34]
    plt.figure(figsize=(10, 6))
    
    # Plot Baseline
    plt.plot(baseline_acc, 'k--', linewidth=2, label='Baseline (Centralized)')
    
    # Plot Exp 1 (Just plotting K=10 and K=50 for clarity)
    if 10 in exp1_results:
        plt.plot(exp1_results[10], label='FedAvg K=10 (IID)')
    if 50 in exp1_results:
        plt.plot(exp1_results[50], label='FedAvg K=50 (IID)')
        
    # Plot Exp 2 (Non-IID)
    if "Non-IID" in exp2_results:
        plt.plot(exp2_results["Non-IID"], label='FedAvg K=10 (Non-IID)')

    plt.title("Work Plan Experiments: Accuracy vs Rounds/Epochs")
    plt.xlabel("Communication Rounds (or Epochs for Baseline)")
    plt.ylabel("Test Accuracy")
    plt.axhline(y=0.9, color='r', linestyle=':', label='Target 90%') # [cite: 38, 44]
    plt.legend()
    plt.grid(True)
    plt.savefig('project_results.png')
    plt.show()
    print("Experiments completed and saved to project_results.png")