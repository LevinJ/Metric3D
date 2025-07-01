"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Tue Jul 01 2025
*  File : test_sampler.py
******************************************* -->

"""

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

# Example datasets
class DatasetA(Dataset):
    def __init__(self):
        self.data = [1, 2, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DatasetB(Dataset):
    def __init__(self):
        self.data = [4, 5, 6]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl")  # Use NCCL for GPU communication
    local_rank = torch.distributed.get_rank()  # Get the rank of the current process
    torch.cuda.set_device(local_rank)  # Set the device for the current process

    # Create datasets
    dataset_a = DatasetA()
    dataset_b = DatasetB()

    # Concatenate datasets
    concat_dataset = ConcatDataset([dataset_a, dataset_b])

    # Create a DistributedSampler
    sampler = DistributedSampler(concat_dataset, shuffle=False)

    # Create a DataLoader with the DistributedSampler
    dataloader = DataLoader(
        concat_dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=2,  # Adjust based on your system
        pin_memory=True,
    )

    # Evaluation loop
    for batch in dataloader:
        print(f"Rank {local_rank}: {batch}")

if __name__ == "__main__":
    main()