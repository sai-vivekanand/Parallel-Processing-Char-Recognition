#code was failed due to nested parallelism error
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
import csv
from datetime import datetime

# ---------------------------
# Configuration Class
# ---------------------------
class Config:
    def __init__(self, cpus, batch_size, epochs, learning_rate=0.001, dataloader_workers=4):
        self.cpus = cpus                    # Number of CPU processes for model parallelism
        self.batch_size = batch_size        # Batch size for training
        self.epochs = epochs                # Number of training epochs
        self.learning_rate = learning_rate  # Learning rate for optimizer
        self.dataloader_workers = dataloader_workers  # Number of worker processes for data loading

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
    # Read CSV file - explicitly specify dtype for the labels column
    df = pd.read_csv(csv_file)
    
    # First column: names; second: labels; remaining: pixel intensities
    names = df.iloc[:, 0]
    
    # Convert labels to strings to ensure uniform dtype
    labels = df.iloc[:, 1].astype(str)  # Convert all labels to strings
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
    print(f"Starting training on partition {partition_id+1}/{config.cpus} with batch size {config.batch_size}")
    
    # Create datasets for this partition
    train_dataset = TMNISTDataset(X_train_part, y_train_part)
    val_dataset = TMNISTDataset(X_val, y_val)
    
    # DataLoaders with appropriate batch sizes and parallel data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.dataloader_workers,  # Parallel data loading
        prefetch_factor=2,                     # Prefetch batches
        pin_memory=True,                       # Pin memory for faster GPU transfer
        persistent_workers=True                # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.dataloader_workers,  # Parallel data loading
        prefetch_factor=2,                     # Prefetch batches
        pin_memory=True                        # Pin memory for faster GPU transfer
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
    print(f"\n=== Training with {config.cpus} CPUs using data parallelism with batch size {config.batch_size} ===")
    
    # Split training data into config.cpus partitions
    indices = np.array_split(range(len(X_train)), config.cpus)
    train_partitions_X = [X_train[idx] for idx in indices]
    train_partitions_y = [y_train[idx] for idx in indices]
    
    print(f"Split data into {config.cpus} partitions of approximate size {len(indices[0])}")
    
    # Train models in parallel using joblib
    start_time = time.time()
    
    # Determine optimal number of dataloader workers (based on available cores)
    # If we have many CPU cores, we can dedicate some to data loading within each process
    total_cores = os.cpu_count() or 4
    cores_per_process = max(1, total_cores // (config.cpus * 2))  # Limit workers to avoid oversubscription
    config.dataloader_workers = min(cores_per_process, 4)  # Cap at 4 workers per dataloader
    
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
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.dataloader_workers,  # Parallel data loading for testing
        pin_memory=True
    )
    test_loss, test_acc = evaluate_model(models[best_model_idx], test_loader)
    print(f"Best individual model test accuracy: {test_acc:.4f}")
    
    # Try ensemble prediction (combine all models)
    ensemble_acc = ensemble_predict(models, test_loader)
    print(f"Ensemble of all {config.cpus} models test accuracy: {ensemble_acc:.4f}")
    
    # Return timing and accuracy metrics
    return {
        "cpus": config.cpus,
        "batch_size": config.batch_size,
        "train_time": total_time,
        "individual_train_times": [result["train_time"] for result in results],
        "best_model_accuracy": test_acc,
        "ensemble_accuracy": ensemble_acc,
        "best_individual_model_idx": best_model_idx,
        "best_val_acc": best_val_acc[best_model_idx],
        "final_train_loss": results[best_model_idx]["train_loss"][-1],
        "final_train_acc": results[best_model_idx]["train_acc"][-1],
        "final_val_loss": results[best_model_idx]["val_loss"][-1],
        "final_val_acc": results[best_model_idx]["val_acc"][-1],
        "individual_results": results
    }

# ---------------------------
# Plot performance metrics and save results
# ---------------------------
def plot_and_save_performance_metrics(all_results, cpu_configs, batch_sizes, base_times):
    # Create a directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    for batch_size in batch_sizes:
        for cpus in cpu_configs:
            result = all_results.get((cpus, batch_size))
            if result:
                base_time = base_times[batch_size]
                speedup = base_time / result["train_time"] if result["train_time"] > 0 else 0
                efficiency = speedup / cpus
                
                # Add to CSV data
                csv_data.append({
                    'batch_size': batch_size,
                    'cpus': cpus,
                    'train_time': result["train_time"],
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'best_model_accuracy': result["best_model_accuracy"],
                    'ensemble_accuracy': result["ensemble_accuracy"],
                    'best_val_acc': result["best_val_acc"],
                    'final_train_loss': result["final_train_loss"],
                    'final_train_acc': result["final_train_acc"],
                    'final_val_loss': result["final_val_loss"],
                    'final_val_acc': result["final_val_acc"]
                })
    
    # Save to CSV
    csv_filename = f'results/joblib_experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Results saved to {csv_filename}")
    
    # Create plots for each batch size
    for batch_size in batch_sizes:
        # Extract data for this batch size
        cpus_for_batch = [cpu for cpu in cpu_configs if (cpu, batch_size) in all_results]
        times = [all_results[(cpu, batch_size)]["train_time"] for cpu in cpus_for_batch]
        speedups = [base_times[batch_size] / t for t in times]
        efficiencies = [speedups[i] / cpus_for_batch[i] for i in range(len(cpus_for_batch))]
        best_accs = [all_results[(cpu, batch_size)]["best_model_accuracy"] for cpu in cpus_for_batch]
        ensemble_accs = [all_results[(cpu, batch_size)]["ensemble_accuracy"] for cpu in cpus_for_batch]
        
        # Batch size specific plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training time
        axes[0, 0].plot(cpus_for_batch, times, 'o-', label=f'Batch Size {batch_size}')
        axes[0, 0].set_xlabel('Number of CPUs')
        axes[0, 0].set_ylabel('Training Time (seconds)')
        axes[0, 0].set_title(f'Training Time vs. CPU Count (Batch Size {batch_size})')
        axes[0, 0].grid(True)
        
        # Speedup
        axes[0, 1].plot(cpus_for_batch, speedups, 'o-', label='Actual')
        axes[0, 1].plot(cpus_for_batch, cpus_for_batch, 'k--', label='Ideal')
        axes[0, 1].set_xlabel('Number of CPUs')
        axes[0, 1].set_ylabel('Speedup')
        axes[0, 1].set_title(f'Speedup vs. CPU Count (Batch Size {batch_size})')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Efficiency
        axes[1, 0].plot(cpus_for_batch, efficiencies, 'o-')
        axes[1, 0].set_xlabel('Number of CPUs')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].set_title(f'Efficiency vs. CPU Count (Batch Size {batch_size})')
        axes[1, 0].grid(True)
        
        # Accuracy
        axes[1, 1].bar(
            np.arange(len(cpus_for_batch)) - 0.2, 
            best_accs, 
            width=0.4, 
            label='Best Individual'
        )
        axes[1, 1].bar(
            np.arange(len(cpus_for_batch)) + 0.2, 
            ensemble_accs, 
            width=0.4, 
            label='Ensemble'
        )
        axes[1, 1].set_xlabel('Number of CPUs')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title(f'Model Accuracy (Batch Size {batch_size})')
        axes[1, 1].set_xticks(np.arange(len(cpus_for_batch)))
        axes[1, 1].set_xticklabels([str(cpu) for cpu in cpus_for_batch])
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y')
        
        plt.tight_layout()
        plt.savefig(f'results/performance_batch{batch_size}.png')
        plt.close()
    
    # Create comparison plots across batch sizes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Different colors for different batch sizes
    colors = ['b', 'r', 'g', 'm']
    markers = ['o', 's', '^', 'D']
    
    for i, batch_size in enumerate(batch_sizes):
        cpus_for_batch = [cpu for cpu in cpu_configs if (cpu, batch_size) in all_results]
        if not cpus_for_batch:
            continue
            
        # Times
        times = [all_results[(cpu, batch_size)]["train_time"] for cpu in cpus_for_batch]
        axes[0, 0].plot(cpus_for_batch, times, f'{colors[i]}{markers[i]}-', label=f'Batch Size {batch_size}')
        
        # Speedups
        speedups = [base_times[batch_size] / t for t in times]
        axes[0, 1].plot(cpus_for_batch, speedups, f'{colors[i]}{markers[i]}-', label=f'Batch Size {batch_size}')
        
        # Efficiency
        efficiencies = [speedups[j] / cpus_for_batch[j] for j in range(len(cpus_for_batch))]
        axes[1, 0].plot(cpus_for_batch, efficiencies, f'{colors[i]}{markers[i]}-', label=f'Batch Size {batch_size}')
        
        # Ensemble accuracy
        ensemble_accs = [all_results[(cpu, batch_size)]["ensemble_accuracy"] for cpu in cpus_for_batch]
        axes[1, 1].plot(cpus_for_batch, ensemble_accs, f'{colors[i]}{markers[i]}-', label=f'Batch Size {batch_size}')
    
    # Add ideal speedup line
    max_cpus = max(cpu_configs)
    axes[0, 1].plot(range(1, max_cpus + 1), range(1, max_cpus + 1), 'k--', label='Ideal')
    
    # Set titles and labels
    axes[0, 0].set_title('Training Time vs. CPU Count')
    axes[0, 0].set_xlabel('Number of CPUs')
    axes[0, 0].set_ylabel('Training Time (seconds)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Speedup vs. CPU Count')
    axes[0, 1].set_xlabel('Number of CPUs')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Efficiency vs. CPU Count')
    axes[1, 0].set_xlabel('Number of CPUs')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Ensemble Accuracy vs. CPU Count')
    axes[1, 1].set_xlabel('Number of CPUs')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/batch_size_comparison.png')
    plt.close()

    # Create individual learning curve plots for each combination
    for batch_size in batch_sizes:
        for cpus in cpu_configs:
            if (cpus, batch_size) not in all_results:
                continue
                
            # Get the best model's metrics
            best_idx = all_results[(cpus, batch_size)]["best_individual_model_idx"]
            metrics = all_results[(cpus, batch_size)]["individual_results"][best_idx]
            
            epochs_range = range(1, len(metrics["train_acc"]) + 1)
            
            # Create the learning curves plot
            plt.figure(figsize=(10, 5))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, metrics["train_acc"], 'b-', label='Train')
            plt.plot(epochs_range, metrics["val_acc"], 'r-', label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Best Model Accuracy (CPUs={cpus}, Batch Size={batch_size})')
            plt.legend()
            plt.grid(True)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, metrics["train_loss"], 'b-', label='Train')
            plt.plot(epochs_range, metrics["val_loss"], 'r-', label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Best Model Loss (CPUs={cpus}, Batch Size={batch_size})')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'results/learning_curves_cpus{cpus}_batch{batch_size}.png')
            plt.close()
    
    print("\nAll performance plots have been saved to the 'results' directory.")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
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
    
    # Define CPU configurations and batch sizes to test
    cpu_configs = [1, 2, 4, 6]  # Test with different numbers of CPUs
    batch_sizes = [32, 64, 128]    # Test with different batch sizes
    epochs = 5
    
    # Store results for each configuration
    all_results = {}
    base_times = {}  # To store baseline times (1 CPU) for each batch size
    
    # Run experiments with each batch size and CPU configuration
    for batch_size in batch_sizes:
        print(f"\n=== Starting experiments with batch size {batch_size} ===")
        
        for num_cpus in cpu_configs:
            config = Config(cpus=num_cpus, batch_size=batch_size, epochs=epochs)
            results = run_data_parallel_training(
                X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, config
            )
            all_results[(num_cpus, batch_size)] = results
            
            # Store baseline time for this batch size
            if num_cpus == 1:
                base_times[batch_size] = results["train_time"]
    
    # Plot results and save to CSV
    plot_and_save_performance_metrics(all_results, cpu_configs, batch_sizes, base_times)
    
    # Print summary of results
    print("\n===== SUMMARY OF RESULTS =====")
    for batch_size in batch_sizes:
        print(f"\n--- BATCH SIZE: {batch_size} ---")
        for cpus in cpu_configs:
            if (cpus, batch_size) not in all_results:
                continue
                
            result = all_results[(cpus, batch_size)]
            base_time = base_times[batch_size]
            speedup = base_time / result["train_time"] if result["train_time"] > 0 else 0
            efficiency = speedup / cpus
            
            print(f"CPUs: {cpus}")
            print(f"  Training time: {result['train_time']:.2f} seconds")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Efficiency: {efficiency:.2f}")
            print(f"  Best model accuracy: {result['best_model_accuracy']:.4f}")
            print(f"  Ensemble accuracy: {result['ensemble_accuracy']:.4f}")
    
    print("\nExperiments completed and results saved!")