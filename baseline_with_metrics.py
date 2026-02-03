import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from datetime import datetime

# --- Neural Network (Unchanged) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# --- Reusable Experiment Tracker ---
class ExperimentTracker:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # Structure: { run_id: { 'train_loss': [], 'test_loss': [], 'test_acc': [] } }
        self.results = {}

    def add_epoch_metric(self, run_id, metric_name, value):
        if run_id not in self.results:
            self.results[run_id] = {'train_loss': [], 'test_loss': [], 'test_acc': []}
        self.results[run_id][metric_name].append(value)

    def save_to_file(self, filename='baseline_metrics.json'):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Metrics saved to {filepath}")

    def plot_aggregated_results(self, title_prefix="Baseline"):
        """Plots mean and std deviation across all runs."""
        epochs = range(1, len(next(iter(self.results.values()))['train_loss']) + 1)
        
        # Aggregate data across runs
        train_losses = np.array([r['train_loss'] for r in self.results.values()])
        test_accuracies = np.array([r['test_acc'] for r in self.results.values()])

        mean_train_loss = np.mean(train_losses, axis=0)
        std_train_loss = np.std(train_losses, axis=0)
        mean_test_acc = np.mean(test_accuracies, axis=0)
        std_test_acc = np.std(test_accuracies, axis=0)

        plt.figure(figsize=(12, 5))

        # Subplot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, mean_train_loss, label='Mean Train Loss', color='blue')
        plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color='blue', alpha=0.2)
        plt.title(f'{title_prefix}: Convergence Speed (Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Subplot 2: Test Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, mean_test_acc, label='Mean Test Acc', color='green')
        plt.fill_between(epochs, mean_test_acc - std_test_acc, mean_test_acc + std_test_acc, color='green', alpha=0.2)
        plt.title(f'{title_prefix}: Final Accuracy & Stability')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f'baseline_plot_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
        plt.show()

# --- Modified Train/Test Functions (Return values instead of just print) ---
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Return average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Train Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}')
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

# --- Main Execution ---
def main():

    # HYPERPARAMETERS:

    parser = argparse.ArgumentParser(description='PyTorch MNIST Baseline')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma')
    parser.add_argument('--no-accel', action='store_true', default=False, help='disables accelerator')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1], help='List of seeds to run for stability check')
    args = parser.parse_args()

    # Device configuration
    if torch.cuda.is_available() and not args.no_accel:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not args.no_accel:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data Loading (Done once)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    
    # Initialize Tracker
    tracker = ExperimentTracker()

    # Loop over seeds to test stability
    for seed in args.seeds:
        print(f"\n--- Starting Run with Seed {seed} ---")
        torch.manual_seed(seed)
        
        train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
        test_kwargs = {'batch_size': 1000}
        
        if device.type == 'cuda':
            cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            # Train and track
            avg_train_loss = train(args, model, device, train_loader, optimizer, epoch)
            tracker.add_epoch_metric(f"seed_{seed}", "train_loss", avg_train_loss)

            # Test and track
            test_loss, test_acc = test(model, device, test_loader)
            tracker.add_epoch_metric(f"seed_{seed}", "test_loss", test_loss)
            tracker.add_epoch_metric(f"seed_{seed}", "test_acc", test_acc)

            scheduler.step()

    # Save and Plot
    tracker.save_to_file()
    tracker.plot_aggregated_results()

if __name__ == '__main__':
    main()