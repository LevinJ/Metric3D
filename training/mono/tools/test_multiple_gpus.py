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
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        

        batch_num = 0
        # Initialize a dictionary to accumulate durations for different code sections
        durations = {
            'data_loading': 0,
            'zero_grad': 0,
            'ddp_model': 0,
            'criterion': 0,
            'backward': 0,
            'optimizer': 0,
            'iteration': 0,
            'dataloader_iterator': 0,
            'inputs_to_device': 0
        }

        # Start timing for dataloader_iterator = iter(train_loader)
        dataloader_iterator_start_time = time.time()
        dataloader_iterator = iter(train_loader)
        dataloader_iterator_end_time = time.time()

        # Accumulate the duration for dataloader_iterator
        durations['dataloader_iterator'] = durations.get('dataloader_iterator', 0) + (dataloader_iterator_end_time - dataloader_iterator_start_time)
        
        while True:
            # Start timing the entire iteration
            iteration_start_time = time.time()
            # Start timing the selected code
            code_start_time = time.time()
            try:
                inputs, labels = next(dataloader_iterator)     
            except StopIteration:
                break
            batch_num += 1
            # End timing the selected code
            code_end_time = time.time()
            # Accumulate the duration for the data loading code
            durations['data_loading'] += (code_end_time - code_start_time)
            
            # Start timing for inputs, labels = inputs.to(rank), labels.to(rank)
            inputs_to_device_start_time = time.time()
            inputs, labels = inputs.to(rank), labels.to(rank)
            inputs_to_device_end_time = time.time()

            # Accumulate the duration for inputs to device
            durations['inputs_to_device'] += (inputs_to_device_end_time - inputs_to_device_start_time)
           

            

            # Print batch size for the first batch of the first epoch
            if epoch == 0 and not hasattr(train_loader, 'printed_flag'):
                print(f"Local rank: {rank}, Batch size: {inputs.size(0)}")
                train_loader.printed_flag = True

            # Start timing optimizer.zero_grad()
            zero_grad_start_time = time.time()
            optimizer.zero_grad()
            # End timing optimizer.zero_grad()
            zero_grad_end_time = time.time()

            # Accumulate the duration for optimizer.zero_grad()
            durations['zero_grad'] += (zero_grad_end_time - zero_grad_start_time)


            # Start timing for outputs = ddp_model(inputs)
            ddp_model_start_time = time.time()
            outputs = ddp_model(inputs)
            ddp_model_end_time = time.time()
            # Accumulate the duration for ddp_model
            durations['ddp_model'] += (ddp_model_end_time - ddp_model_start_time)

            # Start timing for loss = criterion(outputs, labels)
            criterion_start_time = time.time()
            loss = criterion(outputs, labels)
            criterion_end_time = time.time()
            # Accumulate the duration for criterion
            durations['criterion'] += (criterion_end_time - criterion_start_time)

            # Start timing for loss.backward()
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            # Accumulate the duration for backward
            durations['backward'] += (backward_end_time - backward_start_time)

            # Start timing for optimizer.step()
            optimizer_start_time = time.time()
            optimizer.step()
            optimizer_end_time = time.time()
            # Accumulate the duration for optimizer
            durations['optimizer'] += (optimizer_end_time - optimizer_start_time)

            # End timing the entire iteration
            iteration_end_time = time.time()
            # Accumulate the duration for the entire iteration
            durations['iteration'] += (iteration_end_time - iteration_start_time)

        epoch_time = time.time() - epoch_start

        # Print the accumulated durations at the end of the epoch
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | Data loading duration: {durations['data_loading']:.2f}s | Zero grad duration: {durations['zero_grad']:.2f}s | DDP model duration: {durations['ddp_model']:.2f}s | Criterion duration: {durations['criterion']:.2f}s | Backward duration: {durations['backward']:.2f}s | Optimizer duration: {durations['optimizer']:.2f}s | Iteration duration: {durations['iteration']:.2f}s | Dataloader iterator duration: {durations['dataloader_iterator']:.2f}s | Inputs to device duration: {durations['inputs_to_device']:.2f}s")
        print(f"Rank {rank}: Epoch {epoch+1} finished with {batch_num} batches.")
        epoch_times.append(epoch_time)

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
    if world_size > 1:
        world_size = 2  # For testing purposes, limit to 2 GPUs
    num_epochs = 2
    
    print(f"Starting training with {world_size} GPUs...")
    mp.spawn(train, args=(world_size, num_epochs), nprocs=world_size, join=True)