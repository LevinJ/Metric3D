import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '16515'

    # Critical network configuration
    # os.environ["NCCL_IB_DISABLE"] = "1"              # Disable InfiniBand
    # os.environ["NCCL_SOCKET_IFNAME"] = "bond0"       # Force bond0 interface
    # os.environ["NCCL_IB_HCA"] = "bond0"              # Specify HCA
    # os.environ["GLOO_SOCKET_IFNAME"] = "bond0"       # For Gloo backend fallback
    
    # GPU-NUMA affinity
    # NCCL_P2P_LEVEL = os.environ["NCCL_P2P_LEVEL"]
    # print(f"NCCL_P2P_LEVEL for rank {rank}={NCCL_P2P_LEVEL}")
    os.environ["NCCL_P2P_LEVEL"] = "NVL"             # Optimize for cross-NUMA
    # os.environ["NCCL_SHM_DISABLE"] = "1"             # Disable shared memory

    # Set device BEFORE initializing process group
    torch.cuda.set_device(rank)  # Critical addition

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Add this test after init_process_group
    # if rank == 0:
    #     tensor = torch.ones(1).cuda()
    #     for i in range(1, world_size):
    #         dist.send(tensor, i)
    # else:
    #     tensor = torch.zeros(1).cuda()
    #     dist.recv(tensor, 0)
    # print(f"Rank {rank} received {tensor.item()}")

def cleanup():
    dist.destroy_process_group()

def test_dist_barrier(rank, world_size):
    print(f"Process {rank} initializing")
    setup(rank, world_size)

    print(f"Process {rank} reached before barrier")
    dist.barrier()
    print(f"Process {rank} passed barrier")

    cleanup()

def main():
    # Add to setup() before init_process_group:
    os.environ["NCCL_DEBUG"] = "INFO"
    print(torch.cuda.nccl.version()) 
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("This test requires at least 2 GPUs.")
        return

    mp.spawn(test_dist_barrier, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
