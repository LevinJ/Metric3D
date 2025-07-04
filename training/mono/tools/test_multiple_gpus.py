"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Jul 04 2025
*  File : test_multiple_gpus.py
******************************************* -->

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
import time
import os
import numpy as np
from torch.utils.data import DataLoader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, num_epochs=10):
    # Initialize distributed training
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Create model
    model = models.resnet18(num_classes=10)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Distributed sampler setup
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_set, batch_size=256, sampler=train_sampler, num_workers=4
    )
    
    # Timing metrics
    epoch_times = []
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            # Print batch size for the first batch of the first epoch
            if epoch == 0 and not hasattr(train_loader, 'printed_flag'):
                print(f"Local rank: {rank}, Batch size: {inputs.size(0)}")
                train_loader.printed_flag = True
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s')
    
    total_time = time.time() - start_time
    
    # Collect metrics on rank 0
    if rank == 0:
        metrics = {
            'world_size': world_size,
            'total_time': total_time,
            'avg_epoch_time': np.mean(epoch_times),
            'epoch_times': epoch_times
        }
        print(f"\nTraining complete on {world_size} GPUs")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
        
        # Save results to file
        # with open(f'gpu_scaling_{world_size}.txt', 'w') as f:
        #     f.write(str(metrics))
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    num_epochs = 10
    
    print(f"Starting training with {world_size} GPUs...")
    mp.spawn(train, args=(world_size, num_epochs), nprocs=world_size, join=True)