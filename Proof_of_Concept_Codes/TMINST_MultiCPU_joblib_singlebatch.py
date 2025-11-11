import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
from joblib import Parallel, delayed
from tqdm import tqdm

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
    print(f"Loading data from {csv_file}...")
    # Read CSV file; ensure labels are read as strings
    df = pd.read_csv(csv_file)
    # First column: names; second: labels; remaining: pixel intensities
    names = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    pixels = df.iloc[:, 2:].values.astype(np.float32)
    
    # Normalize pixel values to [0, 1]
    pixels /= 255.0
    
    # Map label strings to integer indices and then one-hot encode them
    unique_labels = sorted(labels.unique())
    print(f"Found {len(unique_labels)} unique character classes")
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
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
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = self.fc_layers(x)
        return x

# ---------------------------
# Training Function for a Single Model
# ---------------------------
def train_model(model, train_loader, val_loader, config, device='cpu'):
    criterion = cross_entropy_one_hot
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Containers to track metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            # Convert one-hot target to indices for accuracy computation
            true_labels = torch.argmax(labels, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
        
        # Calculate epoch metrics for training
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == true_labels).sum().item()
        
        # Calculate epoch metrics for validation
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{config.epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "val_loss": val_losses,
        "val_acc": val_accuracies,
        "model": model
    }

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    criterion = cross_entropy_one_hot
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            true_labels = torch.argmax(labels, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            test_total += labels.size(0)
            test_correct += (predicted == true_labels).sum().item()
    
    test_loss /= test_total
    test_acc = test_correct / test_total
    return test_loss, test_acc

# ---------------------------
# Ensemble Prediction Function
# ---------------------------
def ensemble_predict(models, data_loader, device='cpu'):
    all_models_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Sum predictions from all models
            ensemble_outputs = None
            for model in models:
                model.eval()
                outputs = model(inputs)
                if ensemble_outputs is None:
                    ensemble_outputs = outputs
                else:
                    ensemble_outputs += outputs
            
            # Final prediction is based on averaged predictions
            predicted = torch.argmax(ensemble_outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            
            total += labels.size(0)
            all_models_correct += (predicted == true_labels).sum().item()
    
    ensemble_accuracy = all_models_correct / total
    return ensemble_accuracy

# ---------------------------
# Train a model on a dataset partition
# ---------------------------
def train_model_on_partition(partition_id, X_train_part, y_train_part, X_val, y_val, config):
    print(f"Starting training on partition {partition_id+1}/{config.cpus}")
    
    # Create datasets for this partition
    train_dataset = TMNISTDataset(X_train_part, y_train_part)
    val_dataset = TMNISTDataset(X_val, y_val)
    
    # DataLoaders with appropriate batch sizes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        pin_memory=True
    )
    
    # Create and train model
    device = torch.device("cpu")
    num_classes = y_train_part.shape[1]
    model = CNNModel(num_classes).to(device)
    
    train_start_time = time.time()
    results = train_model(model, train_loader, val_loader, config, device)
    train_time = time.time() - train_start_time
    
    results["train_time"] = train_time
    results["partition_id"] = partition_id
    
    return results

# ---------------------------
# Main training execution with data parallelism
# ---------------------------
def run_data_parallel_training(X_train, y_train, X_val, y_val, X_test, y_test, config):
    print(f"\n=== Training with {config.cpus} CPUs using data parallelism ===")
    
    # Split training data into config.cpus partitions
    indices = np.array_split(range(len(X_train)), config.cpus)
    train_partitions_X = [X_train[idx] for idx in indices]
    train_partitions_y = [y_train[idx] for idx in indices]
    
    print(f"Split data into {config.cpus} partitions of approximate size {len(indices[0])}")
    
    # Train models in parallel using joblib
    start_time = time.time()
    
    results = Parallel(n_jobs=config.cpus)(
        delayed(train_model_on_partition)(
            i, train_partitions_X[i], train_partitions_y[i], X_val, y_val, config
        )
        for i in range(config.cpus)
    )
    
    total_time = time.time() - start_time
    print(f"Total training time with {config.cpus} CPUs: {total_time:.2f} seconds")

    # Extract models and find the best one based on validation accuracy
    models = [result["model"] for result in results]
    best_val_acc = [result["val_acc"][-1] for result in results]
    best_model_idx = np.argmax(best_val_acc)
    
    print(f"Best individual model from partition {best_model_idx+1} "
          f"with validation accuracy {best_val_acc[best_model_idx]:.4f}")
    
    # Evaluate best individual model
    test_dataset = TMNISTDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    test_loss, test_acc = evaluate_model(models[best_model_idx], test_loader)
    print(f"Best individual model test accuracy: {test_acc:.4f}")
    
    # Try ensemble prediction (combine all models)
    ensemble_acc = ensemble_predict(models, test_loader)
    print(f"Ensemble of all {config.cpus} models test accuracy: {ensemble_acc:.4f}")
    
    # Return timing and accuracy metrics
    return {
        "cpus": config.cpus,
        "train_time": total_time,
        "individual_train_times": [result["train_time"] for result in results],
        "best_model_accuracy": test_acc,
        "ensemble_accuracy": ensemble_acc,
        "best_individual_model_idx": best_model_idx,
        "individual_results": results
    }

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Load and preprocess the dataset with one-hot label encoding
    csv_file = "../Data/balanced_94_character_TMNIST.csv"
    pixels, one_hot_labels, label_to_idx = load_and_preprocess_data(csv_file)
    
    # Convert one-hot labels back to numeric for stratified splitting
    numeric_labels = np.argmax(one_hot_labels, axis=1)
    
    # Split into train (70%), validation (15%), and test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        pixels, numeric_labels, test_size=0.15, stratify=numeric_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
    )
    
    # Convert back to one-hot encoding
    num_classes = len(label_to_idx)
    y_train_onehot = np.eye(num_classes)[y_train]
    y_val_onehot = np.eye(num_classes)[y_val]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Print dataset shapes
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Define CPU configurations to test
    cpu_configs = [1, 2, 4, 6]  # Test with different numbers of CPUs
    batch_size = 64  # Smaller batch size for more batches & better CPU utilization
    epochs = 5
    
    # Store results for each CPU configuration
    all_results = {}
    
    # Run experiments with each CPU configuration
    for num_cpus in cpu_configs:
        config = Config(cpus=num_cpus, batch_size=batch_size, epochs=epochs)
        results = run_data_parallel_training(
            X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, config
        )
        all_results[num_cpus] = results
    
    # Calculate speedup and efficiency
    base_time = all_results[1]["train_time"]  # Single CPU time as baseline
    
    for cpus in cpu_configs:
        if cpus == 1:
            continue  # Skip baseline
            
        speedup = base_time / all_results[cpus]["train_time"]
        efficiency = speedup / cpus
        
        print(f"\n--- CPU Count: {cpus} ---")
        print(f"Training time: {all_results[cpus]['train_time']:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Efficiency: {efficiency:.2f}")
        print(f"Best model test accuracy: {all_results[cpus]['best_model_accuracy']:.4f}")
        print(f"Ensemble test accuracy: {all_results[cpus]['ensemble_accuracy']:.4f}")
    
    # Plot training time vs CPU count
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    times = [all_results[cpu]["train_time"] for cpu in cpu_configs]
    plt.plot(cpu_configs, times, 'o-')
    plt.xlabel('Number of CPUs')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. CPU Count')
    plt.grid(True)
    
    # Plot speedup vs CPU count
    plt.subplot(1, 3, 2)
    speedups = [base_time / all_results[cpu]["train_time"] for cpu in cpu_configs]
    ideal = cpu_configs.copy()  # Ideal speedup (linear scaling)
    plt.plot(cpu_configs, speedups, 'o-', label='Actual')
    plt.plot(cpu_configs, ideal, '--', label='Ideal')
    plt.xlabel('Number of CPUs')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. CPU Count')
    plt.legend()
    plt.grid(True)
    
    # Plot efficiency vs CPU count
    plt.subplot(1, 3, 3)
    efficiencies = [speedups[i] / cpu_configs[i] for i in range(len(cpu_configs))]
    plt.plot(cpu_configs, efficiencies, 'o-')
    plt.xlabel('Number of CPUs')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs. CPU Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('parallel_performance.png')
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    ind = np.arange(len(cpu_configs))
    width = 0.35
    
    individual_accs = [all_results[cpu]["best_model_accuracy"] for cpu in cpu_configs]
    ensemble_accs = [all_results[cpu]["ensemble_accuracy"] for cpu in cpu_configs]
    
    plt.bar(ind, individual_accs, width, label='Best Individual Model')
    plt.bar(ind + width, ensemble_accs, width, label='Ensemble Model')
    
    plt.xlabel('Number of CPUs')
    plt.ylabel('Test Accuracy')
    plt.title('Model Accuracy vs. CPU Count')
    plt.xticks(ind + width / 2, [str(cpu) for cpu in cpu_configs])
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    
    print("\nPerformance analysis complete. Results saved as 'parallel_performance.png' and 'accuracy_comparison.png'")