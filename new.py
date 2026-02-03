import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- External Dependencies ---
# Ensure 'util.py' and 'data.py' are in the same directory
from util import view_10
from data import fetch_dataset, data_to_tensor, iid_partition_loader, noniid_partition_loader

# =============================================================================
# 1. Setup & Configuration
# =============================================================================

# Set Random Seeds for Reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Set Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"| using device: {device}")

# Hyperparameters for Data Loading
bsz = 10

# Load Data
print("Fetching dataset...")
train_data, test_data = fetch_dataset()

# Global Test Loader (for validation)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

# Federated Partition Loaders
print("Partitioning data for IID and Non-IID clients...")
iid_client_train_loader = iid_partition_loader(train_data, bsz=bsz)
noniid_client_train_loader = noniid_partition_loader(train_data, bsz=bsz)

# Centralized Training Loader (Full Dataset)
central_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)


# =============================================================================
# 2. Model Definition (CNN with Dropout)
# =============================================================================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(1024, 512)
        # Added Dropout Layer with 50% probability
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        # Apply Dropout after activation
        x = self.dropout(x)
        x = self.out(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cnn_test = CNN()
print(f"CNN Initialized. Total Trainable Params: {count_parameters(cnn_test)}")


# =============================================================================
# 3. Training & Validation Helper Functions
# =============================================================================

criterion = nn.CrossEntropyLoss()

def validate(model):
    """Evaluates the model on the global test set."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (t, (x, y)) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.shape[0]
    return correct / total

def train_centralized(model, train_loader, epochs, lr):
    """
    Trains the model in a centralized fashion (Standard DL).
    Returns a list of validation accuracies per epoch.
    """
    print("\n--- Starting Centralized Training Baseline ---")
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    accuracy_history = []
    
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        val_acc = validate(model)
        accuracy_history.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Validation Acc: {val_acc:.4f}")
        
    return np.array(accuracy_history)

def train_client(client_loader, global_model, num_local_epochs, lr):
    """Trains a client locally."""
    # Deep copy global model to ensure we don't modify it in-place
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for epoch in range(num_local_epochs):
        for (i, (x, y)) in enumerate(client_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    return local_model

def running_model_avg(current, next, scale):
    """Aggregates model weights (FedAvg)."""
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current


# =============================================================================
# 4. Federated Experiment Runner
# =============================================================================

def fed_avg_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, experiment_name):
    print(f"\n--- Starting Experiment: {experiment_name} ---")
    round_accuracy = []
    
    # Total clients assuming the loader is a list of all client loaders
    total_clients = len(client_train_loader) 
    
    for t in range(max_rounds):
        # 1. Select Clients
        clients = np.random.choice(np.arange(total_clients), num_clients_per_round, replace=False)
        
        # 2. Prepare Global Model (Move to CPU to save GPU mem during copies)
        global_model.eval()
        global_model = global_model.to('cpu') 
        running_avg = None

        # 3. Train Clients
        for i, c in enumerate(clients):
            # Pass the dataloader for specific client 'c'
            local_model = train_client(client_train_loader[c], global_model, num_local_epochs, lr)
            
            # Aggregate weights
            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1/num_clients_per_round)
        
        # 4. Update Global Model
        global_model.load_state_dict(running_avg)

        # 5. Validate
        val_acc = validate(global_model)
        round_accuracy.append(val_acc)
        
        # Logging
        if t % 10 == 0 or t == max_rounds - 1:
            print(f"Round {t} | Validation Acc: {val_acc:.4f}")

    return np.array(round_accuracy)


# =============================================================================
# 5. Main Execution & Plotting
# =============================================================================

if __name__ == "__main__":
    # Configuration
    LR = 0.01
    LOCAL_EPOCHS = 5
    MAX_ROUNDS = 50       # Communication rounds for FL
    CENTRAL_EPOCHS = 20   # Epochs for Centralized baseline
    RESULTS = {}

    # --- 1. Centralized Baseline ---
    central_model = CNN()
    acc_central = train_centralized(central_model, central_train_loader, CENTRAL_EPOCHS, LR)
    RESULTS['Centralized'] = acc_central

    # --- 2. FL Experiments Configuration ---
    # Format: (num_clients_per_round, data_loader, label)
    experiments = [
        (10, iid_client_train_loader, "FedAvg IID (m=10)"),
        (50, iid_client_train_loader, "FedAvg IID (m=50)"),
        (10, noniid_client_train_loader, "FedAvg Non-IID (m=10)"),
        (50, noniid_client_train_loader, "FedAvg Non-IID (m=50)")
    ]

    for m, loader, label in experiments:
        # Reset model for each experiment
        fl_model = CNN()
        acc = fed_avg_experiment(
            global_model=fl_model, 
            num_clients_per_round=m, 
            num_local_epochs=LOCAL_EPOCHS, 
            lr=LR, 
            client_train_loader=loader, 
            max_rounds=MAX_ROUNDS, 
            experiment_name=label
        )
        RESULTS[label] = acc
        # Save raw data
        np.save(f'{label.replace(" ", "_").lower()}_acc.npy', acc)

    # --- 3. Plotting Results ---
    plt.figure(figsize=(12, 8))

    # Plot Centralized (Stretched to span the graph for baseline comparison)
    plt.plot(RESULTS['Centralized'], linewidth=3, linestyle='--', label='Centralized (Baseline)')

    # Plot FL Experiments
    for key, val in RESULTS.items():
        if key == 'Centralized': continue
        plt.plot(val, label=key)

    plt.title(f'CNN (w/ Dropout) Convergence: Centralized vs Federated (LR={LR}, E={LOCAL_EPOCHS})')
    plt.xlabel('Evaluation Steps (Epochs for Central / Rounds for FL)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig('experiment_comparison.png')
    plt.show()

    print("\nExperiments completed. Plot saved to 'experiment_comparison.png'.")