import torch
import torch.nn.functional as F
import copy
import numpy as np
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps, 
        num_learners=3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        accuracy_type='single', path='log/', finetune=False, finetune_steps=None):
    """
    FedEM: Federated Multi-Task Learning under a Mixture of Distributions
    
    Args:
        dataset_splited: Split dataset for federated learning
        batch_size: Batch size for training
        num_nodes: Number of federated nodes/clients
        model: Model constructor function
        objective: Loss function
        optimizer: Optimizer constructor
        global_rounds: Number of global communication rounds
        local_steps: Number of local training steps per round
        num_learners: Number of learners (models) per client (M in the paper)
        device: Computing device
        accuracy_type: Type of accuracy computation
        path: Path for logging
        finetune: Whether to perform finetuning
        finetune_steps: Number of finetuning steps
    """
    
    # Import server and node classes
    from fedbase.server.server_fl_mot import server_class
    from fedbase.nodes.node_fl_mot import node
    
    # Split dataset
    train_splited, test_splited, split_para = dataset_splited
    
    # Initialize server
    server = server_class(device)
    server.assign_model(model())
    
    # Initialize nodes with multiple learners
    nodes = []
    for i in range(num_nodes):
        node_i = node(i, device)
        
        # data
        node_i.assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        node_i.assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
        
        # model
        node_i.assign_model(model())
        
        # objective
        node_i.assign_objective(objective())
        
        # optim
        node_i.assign_optim(optimizer(node_i.model.parameters()))
        
        # Initialize multiple learners for each node with diverse initialization
        node_i.learners = []
        for m in range(num_learners):
            learner = model().to(device)
            # Apply different initialization strategies for diversity
            with torch.no_grad():
                for name, param in learner.named_parameters():
                    if 'weight' in name and param.dim() > 1:
                        # Use different initialization scales for each learner
                        if m == 0:
                            # Xavier/Glorot uniform (default-like)
                            torch.nn.init.xavier_uniform_(param, gain=1.0)
                        elif m == 1:
                            # Xavier with different gain
                            torch.nn.init.xavier_uniform_(param, gain=0.5)
                        elif m == 2:
                            # He initialization
                            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                        else:
                            # Normal initialization with different std
                            torch.nn.init.normal_(param, mean=0.0, std=0.02 * (m + 1))
                    elif 'bias' in name:
                        # Different bias initialization
                        if m % 2 == 0:
                            torch.nn.init.zeros_(param)
                        else:
                            torch.nn.init.uniform_(param, -0.1, 0.1)
            node_i.learners.append(learner)
        
        node_i.learner_optimizers = [optimizer(learner.parameters()) for learner in node_i.learners]
        
        # Initialize mixing coefficients (π_{i,m})
        node_i.mixing_coeffs = torch.ones(num_learners, device=device) / num_learners
        
        # Initialize assignment probabilities storage
        node_i.assignment_probs = None
        
        nodes.append(node_i)
    
    # Initialize global learners at server with diverse initialization
    global_learners = []
    for m in range(num_learners):
        learner = model().to(device)
        # Apply same diverse initialization as local learners
        with torch.no_grad():
            for name, param in learner.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    if m == 0:
                        torch.nn.init.xavier_uniform_(param, gain=1.0)
                    elif m == 1:
                        torch.nn.init.xavier_uniform_(param, gain=0.5)
                    elif m == 2:
                        torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02 * (m + 1))
                elif 'bias' in name:
                    if m % 2 == 0:
                        torch.nn.init.zeros_(param)
                    else:
                        torch.nn.init.uniform_(param, -0.1, 0.1)
        global_learners.append(learner)
    
    # Copy the diverse global learners to all nodes (instead of overwriting)
    for i in range(num_nodes):
        for m in range(num_learners):
            nodes[i].learners[m].load_state_dict(global_learners[m].state_dict())
    
    # Calculate weight list for aggregation
    weight_list = [nodes[i].data_size / sum([nodes[j].data_size for j in range(num_nodes)]) for i in range(num_nodes)]
    
    print(f"FedEM Training Started: {num_nodes} clients, {num_learners} learners per client, {global_rounds} rounds")
    print(f"Applied diverse initialization to prevent learner collapse")
    
    # Main federated learning loop
    for round_idx in range(global_rounds):
        print(f'-------------------Global round {round_idx} start-------------------')
        
        # Local updates for each node
        for node_idx in range(num_nodes):
            print(f'-------------------Local update {node_idx} start-------------------')
            
            # Perform FedEM local training
            nodes[node_idx].fedem_local_training(local_steps, num_learners)
            
            print(f'-------------------Local update {node_idx} end-------------------')
        
        # Server aggregation: aggregate learners of same type across clients
        for learner_idx in range(num_learners):
            # Collect learners of type learner_idx from all nodes
            learner_models = [nodes[i].learners[learner_idx] for i in range(num_nodes)]
            
            # Aggregate using weighted average
            aggregated_state = server.aggregate(learner_models, weight_list)
            
            # Update global learner
            global_learners[learner_idx].load_state_dict(aggregated_state)
            
            # Distribute back to all nodes
            for i in range(num_nodes):
                nodes[i].learners[learner_idx].load_state_dict(aggregated_state)
        
        # Test accuracy for each node
        for j in range(num_nodes):
            nodes[j].local_test()
        
        # Calculate and print global accuracy
        server.acc(nodes, weight_list)
    
    if finetune and finetune_steps:
        print("Starting FedEM finetuning...")
        for round_idx in range(finetune_steps):
            print(f'-------------------Finetune round {round_idx} start-------------------')
            
            for node_idx in range(num_nodes):
                nodes[node_idx].fedem_local_training(local_steps, num_learners)
            
            # Aggregate and distribute
            for learner_idx in range(num_learners):
                learner_models = [nodes[i].learners[learner_idx] for i in range(num_nodes)]
                aggregated_state = server.aggregate(learner_models, weight_list)
                global_learners[learner_idx].load_state_dict(aggregated_state)
                
                for i in range(num_nodes):
                    nodes[i].learners[learner_idx].load_state_dict(aggregated_state)
            
            for j in range(num_nodes):
                nodes[j].local_test()
            
            server.acc(nodes, weight_list)
    
    print("FedEM Training Completed!")
    return nodes, server 