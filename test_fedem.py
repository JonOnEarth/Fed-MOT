#!/usr/bin/env python3

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from fedbase.utils.data_loader import data_process
from fedbase.model.model import CNNCifar
from fedbase.baselines import fedem

def main():
    """
    Test script for FedEM algorithm
    """
    print("="*60)
    print("FedEM (Federated Multi-Task Learning) Test")
    print("="*60)
    
    # Configuration
    num_nodes = 5
    batch_size = 64
    global_rounds = 5
    local_steps = 5
    num_learners = 3  # Number of learners per client
    
    # Dataset configuration
    domains = ['cifar10']
    client_group = 1
    method = 'iid'
    alpha = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split dataset
    print(f"\nLoading CIFAR-10 dataset...")
    print(f"Configuration: {num_nodes} clients, {num_learners} learners per client")
    print(f"Training: {global_rounds} global rounds, {local_steps} local steps per round")
    
    try:
        # Load CIFAR-10 data
        dataset_loader = data_process('cifar10')
        train_splited, test_splited = dataset_loader.split_dataset(
            num_nodes=num_nodes,
            alpha=alpha,
            method=method
        )
        dataset_splited = train_splited, test_splited, None
        
        print(f"Dataset loaded successfully!")
        print(f"Number of clients: {len(dataset_splited[0])}")
        
        # Print dataset distribution
        for i in range(min(3, num_nodes)):  # Show first 3 clients
            train_size = len(dataset_splited[0][i])
            test_size = len(dataset_splited[1][i])
            print(f"Client {i}: {train_size} training batches, {test_size} test batches")
        
        # Define model, objective, and optimizer
        def model_fn():
            return CNNCifar()
        
        def optimizer_fn(params):
            return optim.Adam(params, lr=0.001)
        
        objective = nn.CrossEntropyLoss
        
        print(f"\nStarting FedEM training...")
        print(f"Model: CNN for CIFAR-10")
        print(f"Optimizer: Adam (lr=0.001)")
        print(f"Objective: CrossEntropyLoss")
        
        # Run FedEM algorithm
        trained_nodes, trained_server = fedem.run(
            dataset_splited=dataset_splited,
            batch_size=batch_size,
            num_nodes=num_nodes,
            model=model_fn,
            objective=objective,
            optimizer=optimizer_fn,
            global_rounds=global_rounds,
            local_steps=local_steps,
            num_learners=num_learners,
            device=device,
            accuracy_type='single'
        )
        
        print(f"\n" + "="*60)
        print("FedEM Training Summary")
        print("="*60)
        
        # Print final results
        if trained_server.test_metrics:
            final_metrics = trained_server.test_metrics[-1]
            print(f"Final Global Accuracy: {100 * final_metrics[0]:.2f}%")
            print(f"Final Global F1 Score: {100 * final_metrics[1]:.2f}%")
        
        # Print individual client results
        print(f"\nFinal Client Results:")
        for i, node in enumerate(trained_nodes):
            if node.test_metrics:
                metrics = node.test_metrics[-1]
                print(f"Client {i}: Accuracy={100 * metrics[0]:.2f}%, F1={100 * metrics[1]:.2f}%")
                print(f"          Mixing coefficients: {[f'{coef:.3f}' for coef in node.mixing_coeffs.cpu().numpy()]}")
        
        print(f"\n" + "="*60)
        print("FedEM Test Completed Successfully!")
        print("="*60)
        
        return trained_nodes, trained_server
        
    except Exception as e:
        print(f"Error during FedEM training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main() 