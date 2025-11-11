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
import os
import pynvml  # For GPU utilization and memory usage

# ---------------------------
# Configuration Class
# ---------------------------
class Config:
    """
    Configuration class to store experiment parameters
    """
    def __init__(self, gpus, batch_size, epochs, learning_rate=0.001):
        self.gpus = gpus          # Number of GPU processes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

# ---------------------------
# Custom Cross Entropy Loss for One-Hot Labels
# ---------------------------
def cross_entropy_one_hot(outputs, targets):
    """
    Custom implementation of cross entropy loss for one-hot encoded labels
    
    Args:
        outputs: Model predictions (logits)
        targets: One-hot encoded target labels
        
    Returns:
        Mean loss across the batch
    """
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(targets * log_probs, dim=1)
    return loss.mean()

# ---------------------------
# Data Preprocessing Functions
# ---------------------------
def load_and_preprocess_data(csv_file):
    """
    Load and preprocess the TMNIST dataset from CSV
    
    Args:
        csv_file: Path to the CSV file containing the dataset
        
    Returns:
        pixels: Normalized pixel values
        one_hot_labels: One-hot encoded labels
        label_to_idx: Mapping from label to index
    """
    df = pd.read_csv(csv_file, dtype={1: str})  # Ensure the labels column is read as string
    names = df.iloc[:, 0]  # First column contains names
    labels = df.iloc[:, 1]  # Second column contains character labels
    pixels = df.iloc[:, 2:].values.astype(np.float32)  # Remaining columns are pixel values
    pixels /= 255.0  # Normalize pixel values to [0,1] range
    
    # Create a mapping from label to index
    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    
    # Convert string labels to numeric indices
    numeric_labels = labels.map(label_to_idx).values
    
    # Create one-hot encoded labels
    num_classes = len(label_to_idx)
    one_hot_labels = np.eye(num_classes)[numeric_labels]
    
    return pixels, one_hot_labels, label_to_idx

# ---------------------------
# Custom Dataset Class for TMNIST
# ---------------------------
class TMNISTDataset(Dataset):
    """
    Custom Dataset class for the TMNIST (Telugu MNIST) dataset
    """
    def __init__(self, pixels, labels):
        """
        Initialize the dataset
        
        Args:
            pixels: Normalized pixel values
            labels: One-hot encoded labels
        """
        self.pixels = pixels
        self.labels = labels  # Expect one-hot encoded labels
        
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: Tensor of shape (1, 28, 28) - single channel 28x28 image
            label: One-hot encoded label tensor
        """
        image = self.pixels[idx].reshape(28, 28)  # Reshape flat pixel array to 28x28 image
        image = np.expand_dims(image, axis=0)  # Add channel dimension: shape: (1, 28, 28)
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ---------------------------
# CNN Architecture Definition
# ---------------------------
class CNNModel(nn.Module):
    """
    Convolutional Neural Network model for character recognition
    """
    def __init__(self, num_classes):
        """
        Initialize the CNN model
        
        Args:
            num_classes: Number of output classes
        """
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 28x28x1, Output: 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input: 14x14x32, Output: 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14x14 -> 7x7
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # Flattened feature maps -> 128 hidden units
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Final classification layer
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps: (batch_size, 64, 7, 7) -> (batch_size, 64*7*7)
        x = self.fc_layers(x)
        return x

# ---------------------------
# Distributed Setup/Cleanup Functions
# ---------------------------
def setup_distributed(rank, world_size):
    """
    Initialize the distributed environment
    
    Args:
        rank: Unique ID for each process
        world_size: Total number of processes (GPUs)
    """
    # Set address for the master process that coordinates all workers
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group with NCCL backend (optimized for CUDA)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()

# ---------------------------
# DDP Training Function
# ---------------------------
def train_ddp(rank, world_size, config, train_dataset, val_dataset, test_dataset, return_dict):
    """
    Train the model using Distributed Data Parallel
    
    Args:
        rank: Unique ID for the current process
        world_size: Total number of processes (GPUs)
        config: Configuration object with hyperparameters
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        return_dict: Shared dictionary to store results
    """
    # Initialize NVML (NVIDIA Management Library) for GPU monitoring
    pynvml.nvmlInit()
    
    # Set up the distributed environment
    setup_distributed(rank, world_size)
    
    # Set the device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Initialize the model and move it to the current GPU
    num_classes = train_dataset.labels.shape[1]
    model = CNNModel(num_classes).to(device)
    
    # Wrap the model with DistributedDataParallel for multi-GPU training
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Define loss function and optimizer
    criterion = cross_entropy_one_hot
    optimizer = optim.Adam(ddp_model.parameters(), lr=config.learning_rate)
    
    # Use DistributedSampler to distribute the data across GPUs
    # Each GPU will get a different subset of the data in each epoch
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders with the distributed samplers
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)
    
    # Initialize lists to record epoch-level metrics
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_train_durations = []       # Epoch training duration (sec)
    epoch_train_throughput = []      # Effective training throughput (samples/sec)
    epoch_gpu_util = []              # GPU utilization (%) per epoch (averaged)
    epoch_mem_util = []              # GPU memory utilization (%) per epoch (averaged)
    
    # Track total training time across all epochs
    training_time_only = 0.0
    
    # Training loop
    for epoch in range(config.epochs):
        # Record the start time of the epoch
        epoch_train_start = time.time()
        
        # Set model to training mode
        ddp_model.train()
        
        # Set epoch for the sampler to ensure different data ordering each epoch
        train_sampler.set_epoch(epoch)
        
        # Initialize metrics for this epoch
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm progress bar only on rank 0 to avoid duplicate outputs
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"GPUs: {world_size} | Epoch {epoch+1}/{config.epochs}")
        else:
            pbar = train_loader
        
        # Training loop for this epoch
        for inputs, labels in pbar:
            # Move data to the current device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients before forward pass
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ddp_model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running metrics
            running_loss += loss.item() * inputs.size(0)
            true_labels = torch.argmax(labels, dim=1)  # Convert one-hot to class indices
            predicted = torch.argmax(outputs, dim=1)   # Get predicted class indices
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
        
        # Record the end time of the epoch and calculate duration
        epoch_train_end = time.time()
        epoch_duration = epoch_train_end - epoch_train_start
        training_time_only += epoch_duration
        epoch_train_durations.append(epoch_duration)
        
        # Calculate throughput (samples per second) accounting for all GPUs
        effective_throughput = (total * world_size) / epoch_duration if epoch_duration > 0 else 0
        epoch_train_throughput.append(effective_throughput)
        
        # Calculate epoch metrics
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation Phase
        ddp_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move data to the current device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = ddp_model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update validation metrics
                val_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == true_labels).sum().item()
        
        # Calculate validation metrics
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Measure GPU utilization and memory usage at the end of the epoch
        try:
            # Get handle for the current GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
            
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            local_gpu_util = float(util.gpu)  # GPU utilization percentage for this device
            
            # Get memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            local_mem_util = (mem_info.used / mem_info.total) * 100.0  # Memory usage percentage
        except Exception as e:
            # Default to zero if measurement fails
            local_gpu_util = 0.0
            local_mem_util = 0.0
        
        # Create a tensor from the local GPU metrics and average it across GPUs
        # This ensures all GPUs have the same values for logging
        gpu_tensor = torch.tensor([local_gpu_util, local_mem_util], device=device, dtype=torch.float32)
        dist.all_reduce(gpu_tensor, op=dist.ReduceOp.SUM)  # Sum values across all processes
        avg_gpu_util = gpu_tensor[0].item() / world_size  # Calculate average
        avg_mem_util = gpu_tensor[1].item() / world_size  # Calculate average
        
        # Record GPU metrics
        epoch_gpu_util.append(avg_gpu_util)
        epoch_mem_util.append(avg_mem_util)
        
        # Only rank 0 prints the progress
        if rank == 0:
            print(f"GPUs: {world_size} | Epoch [{epoch+1}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Epoch Time: {epoch_duration:.2f}s, Throughput: {effective_throughput:.2f} samples/s | "
                  f"GPU Util: {avg_gpu_util:.2f}%, Mem Util: {avg_mem_util:.2f}%")
        
        # Store epoch metrics
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_val_loss.append(val_loss)
        epoch_val_acc.append(val_acc)
    
    # Final evaluation on the test set (only rank 0 performs this)
    if rank == 0:
        # Create a data loader for the test set
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Set model to evaluation mode
        ddp_model.eval()
        
        # Initialize test metrics
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Evaluate on the test set
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move data to the current device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = ddp_model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update test metrics
                test_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == true_labels).sum().item()
        
        # Calculate final test metrics
        test_loss /= test_total
        test_acc = test_correct / test_total
        
        # Print final test results
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Save metrics in the return dictionary for access by the main process
        return_dict[world_size] = {
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "epoch_train_time": epoch_train_durations,
            "train_throughput": epoch_train_throughput,
            "gpu_util": epoch_gpu_util,
            "mem_util": epoch_mem_util,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "training_time_only": training_time_only
        }
    
    # Clean up NVML and distributed resources
    pynvml.nvmlShutdown()
    cleanup_distributed()

# ---------------------------
# Experiment Runner
# ---------------------------
def run_experiment(config, train_dataset, val_dataset, test_dataset):
    """
    Run an experiment with a specific configuration
    
    Args:
        config: Configuration object with hyperparameters
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        
    Returns:
        training_time: Total training time
        metrics: Dictionary of metrics from the experiment
    """
    # Create a shared dictionary to store results across processes
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Spawn multiple processes for distributed training
    mp.spawn(train_ddp,
             args=(config.gpus, config, train_dataset, val_dataset, test_dataset, return_dict),
             nprocs=config.gpus,  # Number of processes to spawn
             join=True)  # Wait for all processes to complete
    
    # Retrieve metrics collected by rank 0
    metrics = return_dict.get(config.gpus, None)
    
    # Extract training time from metrics
    training_time = metrics["training_time_only"] if metrics and "training_time_only" in metrics else None
    
    return training_time, metrics

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Automatically detect available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs available. Exiting.")
        exit(1)
    
    # Print GPU information
    print(f"\nTotal Available GPUs: {available_gpus}")
    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name
        total_mem = props.total_memory / (1024 ** 2)  # bytes to MB
        print(f"GPU {i}: Name: {gpu_name}, Total Memory: {total_mem:.2f} MB")
    
    # Load and preprocess the dataset
    csv_file = "../Data/balanced_94_character_TMNIST.csv"
    pixels, one_hot_labels, label_to_idx = load_and_preprocess_data(csv_file)
    
    # Split into train (70%), validation (15%), and test (15%) using stratification
    # Stratification ensures class distribution is preserved in all splits
    numeric_labels = np.argmax(one_hot_labels, axis=1)  # Convert one-hot to class indices
    
    # First split: 85% train+val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        pixels, numeric_labels, test_size=0.15, stratify=numeric_labels, random_state=42
    )
    
    # Second split: 70% train, 15% val (0.1765 of 85% is ~15% of original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
    )
    
    # Convert class indices back to one-hot encodings
    num_classes = len(label_to_idx)
    y_train_onehot = np.eye(num_classes)[y_train]
    y_val_onehot = np.eye(num_classes)[y_val]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Create dataset objects
    train_dataset = TMNISTDataset(X_train, y_train_onehot)
    val_dataset = TMNISTDataset(X_val, y_val_onehot)
    test_dataset = TMNISTDataset(X_test, y_test_onehot)
    
    # Define the batch sizes and GPU counts to experiment with
    batch_sizes = [64, 128, 256, 512, 1024, 2048]  # Different batch sizes to test
    epochs = 12  # Number of epochs for each experiment
    gpu_counts = list(range(1, available_gpus + 1))  # Test with 1, 2, ..., available_gpus GPUs
    
    # Lists to collect CSV rows for epoch-level and summary test metrics
    epoch_rows = []
    test_rows = []
    
    # Loop over each batch size and GPU configuration
    for batch_size in batch_sizes:
        print(f"\n===== Running experiments for batch size: {batch_size} =====")
        results = {}  # Store experiment results
        times = {}    # Store training times
        
        # Test with different numbers of GPUs
        for gpus in gpu_counts:
            print(f"\n--- Training with {gpus} GPU(s) for batch size {batch_size} ---")
            
            # Create configuration for this experiment
            config = Config(gpus=gpus, batch_size=batch_size, epochs=epochs)
            
            # Run the experiment
            training_time, metrics = run_experiment(config, train_dataset, val_dataset, test_dataset)
            
            # Store results
            times[gpus] = training_time
            results[gpus] = metrics
            
            print(f"Training time with {gpus} GPU(s): {training_time:.2f} seconds")
        
        # Collect detailed metrics for each configuration
        for gpus in gpu_counts:
            # Get GPU information for the GPUs used in this experiment
            gpu_names_list = [torch.cuda.get_device_properties(i).name for i in range(gpus)]
            gpu_mems_list = [f"{torch.cuda.get_device_properties(i).total_memory/(1024**2):.0f}MB" for i in range(gpus)]
            
            # Use unique values and join them with a comma (in case of heterogeneous GPUs)
            gpu_names = ", ".join(sorted(set(gpu_names_list)))
            gpu_mems = ", ".join(sorted(set(gpu_mems_list)))
            
            # Get the metrics for this configuration
            metrics = results[gpus]
            
            # Create a row for each epoch in this configuration
            for epoch_idx in range(epochs):
                row = {
                    'gpu_name': gpu_names,
                    'each_gpu_memory': gpu_mems,
                    'gpu_count': gpus,
                    'batch_size': batch_size,
                    'epoch': epoch_idx + 1,
                    'train_loss': metrics["train_loss"][epoch_idx],
                    'train_acc': metrics["train_acc"][epoch_idx],
                    'val_loss': metrics["val_loss"][epoch_idx],
                    'val_acc': metrics["val_acc"][epoch_idx],
                    'epoch_train_time': metrics["epoch_train_time"][epoch_idx],
                    'train_throughput': metrics["train_throughput"][epoch_idx],
                    'gpu_util': metrics["gpu_util"][epoch_idx],
                    'mem_util': metrics["mem_util"][epoch_idx],
                }
                epoch_rows.append(row)
            
            # Create a summary row for test metrics
            test_summary_row = {
                'gpu_name': gpu_names,
                'each_gpu_memory': gpu_mems,
                'gpu_count': gpus,
                'batch_size': batch_size,
                'total_training_time': metrics["training_time_only"],
                'test_loss': metrics["test_loss"],
                'test_acc': metrics["test_acc"]
            }
            test_rows.append(test_summary_row)
    
    # Save epoch-level metrics to CSV
    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv("DDP_experiment_metrics_epoch.csv", index=False)
    
    # Save final test metrics summary to CSV
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv("DDP_experiment_metrics_test.csv", index=False)
    
    print("All experiments and CSV metric files have been completed and saved.")