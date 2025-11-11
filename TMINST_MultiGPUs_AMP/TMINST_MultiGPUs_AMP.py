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
import pynvml  # For GPU utilization and memory usage monitoring

# ---------------------------
# Configuration Class
# ---------------------------
class Config:
    """
    Configuration class to store experimental parameters.
    This simplifies passing multiple parameters between functions.
    """
    def __init__(self, gpus, batch_size, epochs, learning_rate=0.001):
        self.gpus = gpus          # Number of GPU processes to use
        self.batch_size = batch_size  # Batch size for each GPU
        self.epochs = epochs      # Number of training epochs
        self.learning_rate = learning_rate  # Initial learning rate for optimizer

# ---------------------------
# Custom Cross Entropy Loss for One-Hot Labels
# ---------------------------
def cross_entropy_one_hot(outputs, targets):
    """
    Custom cross-entropy loss for one-hot encoded labels.
    Standard cross_entropy in PyTorch expects class indices, not one-hot vectors.
    
    Args:
        outputs: Model predictions (logits)
        targets: One-hot encoded true labels
        
    Returns:
        Mean loss value across the batch
    """
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(targets * log_probs, dim=1)
    return loss.mean()

# ---------------------------
# Data Preprocessing Functions
# ---------------------------
def load_and_preprocess_data(csv_file):
    """
    Load and preprocess TMNIST data from CSV file.
    
    Args:
        csv_file: Path to the CSV file containing TMNIST data
        
    Returns:
        pixels: Normalized pixel values
        one_hot_labels: One-hot encoded class labels
        label_to_idx: Dictionary mapping class labels to indices
    """
    df = pd.read_csv(csv_file, dtype={1: str})  # Force second column (labels) to be string type
    names = df.iloc[:, 0]  # First column contains names (not used in this script)
    labels = df.iloc[:, 1]  # Second column contains class labels
    pixels = df.iloc[:, 2:].values.astype(np.float32)  # All remaining columns are pixel values
    pixels /= 255.0  # Normalize pixel values to [0, 1] range
    
    # Create mapping from text labels to numeric indices
    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    numeric_labels = labels.map(label_to_idx).values
    
    # Convert numeric labels to one-hot encoding
    num_classes = len(label_to_idx)
    one_hot_labels = np.eye(num_classes)[numeric_labels]
    return pixels, one_hot_labels, label_to_idx

# ---------------------------
# Custom Dataset Class for TMNIST
# ---------------------------
class TMNISTDataset(Dataset):
    """
    PyTorch Dataset implementation for TMNIST dataset.
    Prepares data for consumption by DataLoader.
    """
    def __init__(self, pixels, labels):
        """
        Initialize dataset with pixel data and labels.
        
        Args:
            pixels: Preprocessed pixel values
            labels: One-hot encoded class labels
        """
        self.pixels = pixels
        self.labels = labels  # Expect one-hot encoded labels
        
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Reshapes flat pixel array to 2D image and adds channel dimension.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            image tensor and corresponding label tensor
        """
        image = self.pixels[idx].reshape(28, 28)  # Reshape to 28x28 image
        image = np.expand_dims(image, axis=0)  # Add channel dimension: shape becomes (1, 28, 28)
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ---------------------------
# CNN Architecture Definition
# ---------------------------
class CNNModel(nn.Module):
    """
    Convolutional Neural Network model for image classification.
    Simple architecture with two convolutional layers followed by two fully connected layers.
    """
    def __init__(self, num_classes):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes for classification
        """
        super(CNNModel, self).__init__()
        # Convolutional layers with pooling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 1x28x28, Output: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input: 32x14x14, Output: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14x14 -> 7x7
        )
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # Flatten and reduce dimensions
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Final output layer
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps to (batch_size, 64*7*7)
        x = self.fc_layers(x)
        return x

# ---------------------------
# Distributed Setup/Cleanup Functions
# ---------------------------
def setup_distributed(rank, world_size):
    """
    Initialize the distributed training environment.
    
    Args:
        rank: Unique ID of the current process
        world_size: Total number of processes participating
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of the machine that will host the process with rank 0
    os.environ['MASTER_PORT'] = '29500'      # Port for coordination
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # Use NCCL backend for CUDA operations

def cleanup_distributed():
    """Clean up the distributed environment after training"""
    dist.destroy_process_group()

# ---------------------------
# DDP Training Function with Mixed Precision
# ---------------------------
def train_ddp(rank, world_size, config, train_dataset, val_dataset, test_dataset, return_dict):
    """
    Main distributed training function executed by each process.
    
    Args:
        rank: Unique ID of the current process
        world_size: Total number of processes participating
        config: Configuration object with hyperparameters
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        test_dataset: Dataset for final testing
        return_dict: Shared dictionary to return metrics to the parent process
    """
    # Initialize NVML for GPU monitoring (each process initializes its own copy)
    pynvml.nvmlInit()
    
    # Set up the distributed environment
    setup_distributed(rank, world_size)
    
    # Set the device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Initialize the model, wrap it with DDP, and prepare optimizer
    num_classes = train_dataset.labels.shape[1]
    model = CNNModel(num_classes).to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)  # Wrap model in DDP
    criterion = cross_entropy_one_hot
    optimizer = optim.Adam(ddp_model.parameters(), lr=config.learning_rate)
    
    # Initialize the GradScaler for AMP mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # Use DistributedSampler to partition the data across processes
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
    
    training_time_only = 0.0  # Total time spent in training (excluding validation)
    
    # Main training loop
    for epoch in range(config.epochs):
        epoch_train_start = time.time()
        
        # Training phase
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # Set epoch for the sampler to reshuffle data
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm progress bar only on the master process (rank 0)
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"GPUs: {world_size} | Epoch {epoch+1}/{config.epochs}")
        else:
            pbar = train_loader
        
        # Process each batch
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            
            # Mixed precision forward pass using bfloat16 autocast for H100 GPUs
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale loss and perform backward pass with gradient scaling (for mixed precision)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate statistics
            running_loss += loss.item() * inputs.size(0)
            true_labels = torch.argmax(labels, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
        
        # Calculate epoch training time and throughput
        epoch_train_end = time.time()
        epoch_duration = epoch_train_end - epoch_train_start
        training_time_only += epoch_duration
        epoch_train_durations.append(epoch_duration)
        
        # Calculate effective throughput (samples processed per second across all GPUs)
        effective_throughput = (total * world_size) / epoch_duration if epoch_duration > 0 else 0
        epoch_train_throughput.append(effective_throughput)
        
        # Calculate epoch-level training metrics
        train_loss = running_loss / total
        train_acc = correct / total
        
        # ---------------------------
        # Validation Phase
        # ---------------------------
        ddp_model.eval()  # Switch model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradients for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == true_labels).sum().item()
        
        # Calculate validation metrics
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # ---------------------------
        # GPU Utilization & Memory Measurement
        # ---------------------------
        try:
            # Get handle to the GPU for this process
            handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            local_gpu_util = float(util.gpu)  # GPU utilization percentage for this device
            # Get memory utilization
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            local_mem_util = (mem_info.used / mem_info.total) * 100.0  # Memory usage percentage
        except Exception:
            # Handle cases where NVML might fail
            local_gpu_util = 0.0
            local_mem_util = 0.0
        
        # Synchronize GPU metrics across all processes using all_reduce
        gpu_tensor = torch.tensor([local_gpu_util, local_mem_util], device=device, dtype=torch.float32)
        dist.all_reduce(gpu_tensor, op=dist.ReduceOp.SUM)
        avg_gpu_util = gpu_tensor[0].item() / world_size
        avg_mem_util = gpu_tensor[1].item() / world_size
        
        # Store GPU utilization metrics
        epoch_gpu_util.append(avg_gpu_util)
        epoch_mem_util.append(avg_mem_util)
        
        # Print epoch results on the master process only
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
    
    # ---------------------------
    # Final Test Evaluation (Rank 0 Only)
    # ---------------------------
    if rank == 0:
        # Only master process evaluates on test set to avoid duplicate work
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        ddp_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                true_labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == true_labels).sum().item()
        
        # Calculate and print final test metrics
        test_loss /= test_total
        test_acc = test_correct / test_total
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Store all metrics in the shared dictionary for the parent process
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
    Run a single experiment with the given configuration.
    Spawns multiple processes for distributed training.
    
    Args:
        config: Configuration object with hyperparameters
        train_dataset, val_dataset, test_dataset: Datasets for training, validation and testing
        
    Returns:
        training_time: Total time spent in training
        metrics: Dictionary of performance metrics
    """
    # Create a managed dictionary to share results between processes
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Spawn multiple processes to handle distributed training
    mp.spawn(train_ddp,
             args=(config.gpus, config, train_dataset, val_dataset, test_dataset, return_dict),
             nprocs=config.gpus,
             join=True)
    
    # Extract results from the shared dictionary
    metrics = return_dict.get(config.gpus, None)
    training_time = metrics["training_time_only"] if metrics and "training_time_only" in metrics else None
    return training_time, metrics

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Auto-detect available GPUs and their properties
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs available. Exiting.")
        exit(1)
    
    # Print information about available GPUs
    print(f"\nTotal Available GPUs: {available_gpus}")
    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name
        total_mem = props.total_memory / (1024 ** 2)  # Convert bytes to MB
        print(f"GPU {i}: Name: {gpu_name}, Total Memory: {total_mem:.2f} MB")
    
    # Load and preprocess the dataset
    csv_file = "../Data/balanced_94_character_TMNIST.csv"
    pixels, one_hot_labels, label_to_idx = load_and_preprocess_data(csv_file)
    
    # Split data into train (70%), validation (15%), and test (15%) using stratification
    # First split off test set
    numeric_labels = np.argmax(one_hot_labels, axis=1)
    X_temp, X_test, y_temp, y_test = train_test_split(pixels, numeric_labels, test_size=0.15, stratify=numeric_labels, random_state=42)
    # Then split remaining data into train and validation
    # Note: 0.1765 of the original 85% is ~15% of the total, giving a 70/15/15 split
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
    
    # Convert labels back to one-hot encoding
    num_classes = len(label_to_idx)
    y_train_onehot = np.eye(num_classes)[y_train]
    y_val_onehot = np.eye(num_classes)[y_val]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Create PyTorch datasets
    train_dataset = TMNISTDataset(X_train, y_train_onehot)
    val_dataset = TMNISTDataset(X_val, y_val_onehot)
    test_dataset = TMNISTDataset(X_test, y_test_onehot)
    
    # Define the experimental parameters to iterate over
    batch_sizes = [64, 128, 256, 512, 1024, 2048]  # Batch sizes to test
    epochs = 12  # Number of epochs for each experiment
    gpu_counts = list(range(1, available_gpus + 1))  # Test from 1 GPU up to all available
    
    # Lists to collect data for CSV output
    epoch_rows = []  # Detailed per-epoch metrics
    test_rows = []   # Summary metrics for each experiment
    
    # Iterate over all batch sizes
    for batch_size in batch_sizes:
        print(f"\n===== Running experiments for batch size: {batch_size} =====")
        results = {}  # Store metrics from each experiment
        times = {}    # Store training times
        
        # Iterate over all GPU counts
        for gpus in gpu_counts:
            print(f"\n--- Training with {gpus} GPU(s) for batch size {batch_size} ---")
            
            # Create configuration for this experiment
            config = Config(gpus=gpus, batch_size=batch_size, epochs=epochs)
            
            # Run the experiment and get results
            training_time, metrics = run_experiment(config, train_dataset, val_dataset, test_dataset)
            times[gpus] = training_time
            results[gpus] = metrics
            print(f"Training time with {gpus} GPU(s): {training_time:.2f} seconds")
        
        # Process and save results for each GPU configuration
        for gpus in gpu_counts:
            # Collect GPU hardware information
            gpu_names_list = [torch.cuda.get_device_properties(i).name for i in range(gpus)]
            gpu_mems_list = [f"{torch.cuda.get_device_properties(i).total_memory/(1024**2):.0f}MB" for i in range(gpus)]
            gpu_names = ", ".join(sorted(set(gpu_names_list)))  # Unique GPU names
            gpu_mems = ", ".join(sorted(set(gpu_mems_list)))    # Unique GPU memory sizes
            
            # Extract metrics for this experiment
            metrics = results[gpus]
            
            # Record per-epoch metrics
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
            
            # Record summary metrics for the full experiment
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
    
    # Save all collected metrics to CSV files
    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv("AMP_experiment_metrics_epoch.csv", index=False)
    
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv("AMP_experiment_metrics_test.csv", index=False)
    
    print("All experiments and CSV metric files have been completed and saved.")