import os
import os.path as osp
import sys
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from tqdm import tqdm  # Add tqdm for progress bar


CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.utils.logger import setup_logger
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.datasets.distributed_sampler import build_dataset_n_sampler_with_cfg
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.db import load_data_info
from mono.model.criterion import build_criterions

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Evaluation')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--load-from', help='Checkpoint file to load weights from')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm', help='job launcher')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args()

    
    from pathlib import Path
    metric3d_dir = Path(__file__).resolve().parents[3]

    args.launcher = 'None'
    args.config =  '/home/levin/workspace/nerf/tools/Metric3D/training/mono/configs/RAFTDecoder/vit.raft5.large.kitti.py'
    args.load_from = f'{metric3d_dir}/weights/metric_depth_vit_large_800k.pth'
   

    # args.test_data_path = f'{args.db_root}/eigen_test.json'
    import time
    args.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.batchsize_per_gpu = 8
    return args
def to_cuda(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data
def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    cfg.dist_params.nnodes = args.nnodes
    cfg.dist_params.node_rank = args.node_rank
    cfg.batchsize_per_gpu = args.batchsize_per_gpu
    cfg.load_from = args.load_from
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if hasattr(args, 'work_dir') and args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0], args.timestamp)

    # create work_dir
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    # log file
    cfg.log_file = osp.join(cfg.work_dir, f'{args.timestamp}.log')

    # load data info
    data_info = {}
    load_data_info('data_server_info', data_info=data_info)
    cfg.db_info = data_info

    # distributed setup
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
    init_env(args.launcher, cfg)
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher)
    else:
        mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher))

def main_worker(local_rank, cfg, launcher):
    logger = setup_logger(cfg.get('log_file', f'eval_{local_rank}.log'))
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=cfg.dist_params.backend,
            init_method=cfg.dist_params.dist_url,
            world_size=cfg.dist_params.world_size,
            rank=cfg.dist_params.global_rank,
            timeout=timedelta(minutes=30)
        )
    # Build dataset and sampler
    val_dataset, val_sampler = build_dataset_n_sampler_with_cfg(cfg, 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batchsize_per_gpu,
        sampler=val_sampler,
        num_workers=cfg.get('thread_per_gpu', 4),
        pin_memory=True,
        drop_last=False
    )
    # Build model
    # build criterions
    criterions = build_criterions(cfg)
    
    # build model
    model = get_configured_monodepth_model(cfg,criterions) 
    
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    model, _, _, _ = load_ckpt(cfg.load_from, model, optimizer=None, scheduler=None, strict_match=False, loss_scaler=None)
    model.eval()
    # Metric
    dam = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    with torch.no_grad():
        # Wrap val_loader with tqdm for progress bar
        for i, batch in enumerate(tqdm(val_loader, desc="Validating", leave=True)):
            # Move batch to GPU
            data = {k: v.cuda(local_rank, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            # data = to_cuda(data)
            output = model.module.inference(data)
            pred_depth = output['prediction']
            gt_depth = data['target']
            pad = data['pad'].squeeze()
            assert torch.allclose(pad[0], pad[-1]), "pad[0] and pad[-1] are not approximately equal"
            pad = pad[0]
            B, C, H, W = pred_depth.shape
            pred_depth = pred_depth[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            gt_depth = gt_depth[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            mask = gt_depth > 0
            dam.update_metrics_gpu(pred_depth, gt_depth, mask, cfg.distributed)
            # Print memory usage
            # print(f"Batch {i}: Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            # Free memory
            del output, pred_depth, gt_depth, mask, data, batch
            torch.cuda.empty_cache()
    # Reduce metrics across all processes
    if local_rank == 0:
        eval_error = dam.get_metrics()
        print('w/o match :', eval_error)
   
    if cfg.distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    main(args)
