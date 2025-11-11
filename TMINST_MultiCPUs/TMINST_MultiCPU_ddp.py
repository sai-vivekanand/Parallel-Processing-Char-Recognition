import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

# ---------------------------
# Configuration Class
# ---------------------------
class Config:
    def __init__(self, cpus, batch_size, epochs, learning_rate=0.001):
        self.cpus = cpus          # Number of CPU processes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

# ---------------------------
# Custom Cross Entropy Loss for One-Hot Labels
# ---------------------------
def cross_entropy_one_hot(outputs, targets):
    # outputs: (batch_size, num_classes)
    # targets: (batch_size, num_classes) as one-hot vectors
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(targets * log_probs, dim=1)
    return loss.mean()

# ---------------------------
# Data Preprocessing Functions
# ---------------------------
def load_and_preprocess_data(csv_file):
    # Read CSV file; ensure labels are read as strings
    df = pd.read_csv(csv_file, dtype={1: str})
    # First column: names; second: labels; remaining: pixel intensities
    names = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    pixels = df.iloc[:, 2:].values.astype(np.float32)
    # Normalize pixel values to [0, 1]
    pixels /= 255.0
    # Map label strings to integer indices and then one-hot encode them
    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    numeric_labels = labels.map(label_to_idx).values
    num_classes = len(label_to_idx)
    one_hot_labels = np.eye(num_classes)[numeric_labels]
    return pixels, one_hot_labels, label_to_idx

# ---------------------------
# Custom Dataset Class for TMNIST
# ---------------------------
class TMNISTDataset(Dataset):
    def __init__(self, pixels, labels):
        self.pixels = pixels
        self.labels = labels  # Expect one-hot encoded labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Reshape flattened vector into a 28x28 image and add channel dimension
        image = self.pixels[idx].reshape(28, 28)
        image = np.expand_dims(image, axis=0)  # shape: (1, 28, 28)
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ---------------------------
# CNN Architecture Definition
# ---------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = self.fc_layers(x)
        return x

# ---------------------------
# Distributed Setup/Cleanup Functions
# ---------------------------
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

# ---------------------------
# DDP Training Function
# ---------------------------
def train_ddp(rank, world_size, config, train_dataset, val_dataset, test_dataset, return_dict):
    setup_distributed(rank, world_size)
    device = torch.device("cpu")
    
    # Initialize model, custom loss function, optimizer and wrap model in DDP
    num_classes = train_dataset.labels.shape[1]
    model = CNNModel(num_classes).to(device)
    ddp_model = DDP(model)
    criterion = cross_entropy_one_hot
    optimizer = optim.Adam(ddp_model.parameters(), lr=config.learning_rate)
    
    # Create Distributed Samplers and DataLoaders for train and validation sets
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)
    
    # Containers to store epoch metrics
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    
    # Variable to accumulate training-only time (excluding validation/test)
    training_time_only = 0.0
    
    for epoch in range(config.epochs):
        # Measure training-only time for each epoch
        epoch_train_start = time.time()
        
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"CPUs: {world_size} | Epoch {epoch+1}/{config.epochs}")
        else:
            pbar = train_loader
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            # Convert one-hot target to indices for accuracy computation
            true_labels = torch.argmax(labels, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
        
        # End of training loop for this epoch
        epoch_train_end = time.time()
        epoch_training_duration = epoch_train_end - epoch_train_start
        training_time_only += epoch_training_duration
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation phase (timed separately, not included in training time)
        ddp_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == true_labels).sum().item()
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        if rank == 0:
            print(f"CPUs: {world_size} | Epoch [{epoch+1}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_val_loss.append(val_loss)
        epoch_val_acc.append(val_acc)
    
    # Final evaluation on the test set (only rank 0 does this)
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        ddp_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == true_labels).sum().item()
        test_loss /= test_total
        test_acc = test_correct / test_total
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        return_dict[world_size] = {
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "training_time_only": training_time_only
        }
    
    cleanup_distributed()

# ---------------------------
# Experiment Runner
# ---------------------------
def run_experiment(config, train_dataset, val_dataset, test_dataset):
    manager = mp.Manager()
    return_dict = manager.dict()
    # Spawn processes equal to the number of CPUs specified
    mp.spawn(train_ddp, args=(config.cpus, config, train_dataset, val_dataset, test_dataset, return_dict), nprocs=config.cpus, join=True)
    metrics = return_dict.get(config.cpus, None)  # Retrieve metrics aggregated by rank 0
    training_time = metrics["training_time_only"] if metrics and "training_time_only" in metrics else None
    return training_time, metrics

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Load and preprocess the dataset with one-hot label encoding
    csv_file = "../Data/balanced_94_character_TMNIST.csv"
    pixels, one_hot_labels, label_to_idx = load_and_preprocess_data(csv_file)
    
    # Split into train (70%), validation (15%), and test (15%) using stratification based on integer labels
    # For stratification, convert one-hot back to integer indices using argmax along axis=1.
    numeric_labels = np.argmax(one_hot_labels, axis=1)
    X_temp, X_test, y_temp, y_test = train_test_split(pixels, numeric_labels, test_size=0.15, stratify=numeric_labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
    
    # Convert integer labels back to one-hot encoding for the datasets
    num_classes = len(label_to_idx)
    y_train_onehot = np.eye(num_classes)[y_train]
    y_val_onehot = np.eye(num_classes)[y_val]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    train_dataset = TMNISTDataset(X_train, y_train_onehot)
    val_dataset = TMNISTDataset(X_val, y_val_onehot)
    test_dataset = TMNISTDataset(X_test, y_test_onehot)
    
    # List of CPU counts to experiment with
    cpu_list = [1, 2, 4, 6]
    batch_size = 512
    epochs = 5  # Adjust number of epochs as needed
    results = {}
    times = {}
    
    # Run experiments for each CPU configuration
    for cpus in cpu_list:
        print(f"\n=== Training with {cpus} CPUs ===")
        config = Config(cpus=cpus, batch_size=batch_size, epochs=epochs)
        training_time, metrics = run_experiment(config, train_dataset, val_dataset, test_dataset)
        times[cpus] = training_time
        results[cpus] = metrics
        print(f"Training time with {cpus} CPUs: {training_time:.2f} seconds")
    
    # ---------------------------
    # Performance Analysis
    # ---------------------------
    # Calculate speedup and efficiency relative to the baseline (lowest CPU count)
    baseline_time = times[cpu_list[0]]
    speedups = {cpus: baseline_time / times[cpus] for cpus in cpu_list}
    efficiencies = {cpus: speedups[cpus] / (cpus / cpu_list[0]) for cpus in cpu_list}
    
    # Plot: Training Time vs. CPU Count
    plt.figure()
    plt.plot(list(times.keys()), list(times.values()), marker='o')
    plt.xlabel("CPU Count")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs. CPU Count")
    plt.savefig("training_time_vs_cpu.png")
    
    # Plot: Speedup vs. CPU Count
    plt.figure()
    plt.plot(list(speedups.keys()), list(speedups.values()), marker='o')
    plt.xlabel("CPU Count")
    plt.ylabel("Speedup")
    plt.title("Speedup vs. CPU Count")
    plt.savefig("DDP_CPU_speedup_vs_cpu.png")
    
    # Plot: Efficiency vs. CPU Count
    plt.figure()
    plt.plot(list(efficiencies.keys()), list(efficiencies.values()), marker='o')
    plt.xlabel("CPU Count")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs. CPU Count")
    plt.savefig("DDP_CPU_efficiency_vs_cpu.png")
    
    # Save training and validation accuracy/loss curves for each CPU configuration
    for cpus in cpu_list:
        metrics = results[cpus]
        epochs_range = range(1, epochs + 1)
        
        # Accuracy Curves
        plt.figure()
        plt.plot(epochs_range, metrics["train_acc"], label="Train Accuracy")
        plt.plot(epochs_range, metrics["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curves (CPUs = {cpus})")
        plt.legend()
        plt.savefig(f"accuracy_curves_{cpus}.png")
        
        # Loss Curves
        plt.figure()
        plt.plot(epochs_range, metrics["train_loss"], label="Train Loss")
        plt.plot(epochs_range, metrics["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves (CPUs = {cpus})")
        plt.legend()
        plt.savefig(f"loss_curves_{cpus}.png")
    
    print("Performance plots have been saved.")