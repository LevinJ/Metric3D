import os
import torch
import matplotlib.pyplot as plt
from mono.model.monodepth_model import get_configured_monodepth_model
from tensorboardX import SummaryWriter
from mono.utils.comm import TrainingStats
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.running import build_lr_schedule_with_cfg, build_optimizer_with_cfg, load_ckpt, save_ckpt
from mono.utils.comm import reduce_dict, main_process, get_rank
from mono.utils.visualization import save_val_imgs, visual_train_data, create_html, save_normal_val_imgs
import traceback
from mono.utils.visualization import create_dir_for_validate_meta
from mono.model.criterion import build_criterions
from mono.datasets.distributed_sampler import build_dataset_n_sampler_with_cfg, build_data_array
from mono.utils.logger import setup_logger
import logging
from .misc import NativeScalerWithGradNormCount, is_bf16_supported
import math
import sys
import random
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
import torch.autograd.profiler

def to_cuda(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def do_train(local_rank: int, cfg: dict):

    logger = logging.getLogger()

    # build criterions
    criterions = build_criterions(cfg)
    
    # build model
    model = get_configured_monodepth_model(cfg,
                                           criterions,
                                           ) 
    
    # log model state_dict
    if main_process():
        logger.info(model.state_dict().keys())
    
    # build datasets
    train_dataset, train_sampler = build_dataset_n_sampler_with_cfg(cfg, 'train')
    if 'multi_dataset_eval' in cfg.evaluation and cfg.evaluation.multi_dataset_eval:
        val_dataset = build_data_array(cfg, 'val')
    else:
        val_dataset, val_sampler = build_dataset_n_sampler_with_cfg(cfg, 'val')
    # build data loaders
    g = torch.Generator()
    g.manual_seed(cfg.seed + cfg.dist_params.global_rank)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.batchsize_per_gpu,
                                                   num_workers=cfg.thread_per_gpu,
                                                   sampler=train_sampler,
                                                   drop_last=True, 
                                                   pin_memory=True,
                                                   generator=g,)
                                                #    collate_fn=collate_fn)
    if isinstance(val_dataset, list):
        val_dataloader = [torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=1,
                                                      num_workers=0,
                                                      sampler=torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False),
                                                      drop_last=True,
                                                      pin_memory=True,) for val_group in val_dataset for val_dataset in val_group]
    else:
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=1,
                                                num_workers=0,
                                                sampler=val_sampler,
                                                drop_last=True,
                                                pin_memory=True,)
    
    # build schedule
    lr_scheduler = build_lr_schedule_with_cfg(cfg)
    optimizer = build_optimizer_with_cfg(cfg, model)
   
    # config distributed training
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), 
                                                          device_ids=[local_rank], 
                                                          output_device=local_rank, 
                                                          find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model.cuda())
    
    # init automatic mix precision training
    # if 'AMP' in cfg.runner.type:
    #     loss_scaler = NativeScalerWithGradNormCount()
    # else:
    #     loss_scaler = None
    loss_scaler = None
    
    # load ckpt
    if cfg.load_from and cfg.resume_from is None:
        model, _, _, loss_scaler = load_ckpt(cfg.load_from, model, optimizer=None, scheduler=None, strict_match=False, loss_scaler=loss_scaler)
    elif cfg.resume_from:
        model, optimizer, lr_scheduler, loss_scaler = load_ckpt(
            cfg.resume_from, 
            model, 
            optimizer=optimizer, 
            scheduler=lr_scheduler, 
            strict_match=False, 
            loss_scaler=loss_scaler)

    if cfg.runner.type == 'IterBasedRunner':
        train_by_iters(cfg,
                    model, 
                    optimizer, 
                    lr_scheduler,
                    train_dataloader,
                    val_dataloader,
                    )
    elif cfg.runner.type == 'IterBasedRunner_MultiSize':
        train_by_iters_multisize(cfg,
                    model, 
                    optimizer, 
                    lr_scheduler,
                    train_dataloader,
                    val_dataloader,
                    )
    elif cfg.runner.type == 'IterBasedRunner_AMP':
        train_by_iters_amp(
            cfg = cfg,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_scaler=loss_scaler
        )
    elif cfg.runner.type == 'IterBasedRunner_AMP_MultiSize':
        train_by_iters_amp_multisize(
            cfg = cfg,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_scaler=loss_scaler
        )
    elif cfg.runner.type == 'EpochBasedRunner':
        raise RuntimeError('It is not supported currently. :)')
    else:
        raise RuntimeError('It is not supported currently. :)')


def train_by_iters(cfg, model, optimizer, lr_scheduler, train_dataloader, val_dataloader):
    """
    Do the training by iterations.
    """
    logger = logging.getLogger()
    tb_logger = None
    if cfg.use_tensorboard and main_process():
        tb_logger = SummaryWriter(cfg.tensorboard_dir)
    if main_process():
        training_stats = TrainingStats(log_period=cfg.log_interval, tensorboard_logger=tb_logger)
    
    lr_scheduler.before_run(optimizer)
    
    # set training steps
    max_iters = cfg.runner.max_iters
    start_iter = lr_scheduler._step_count

    save_interval = cfg.checkpoint_config.interval
    eval_interval = cfg.evaluation.interval
    epoch = 0
    logger.info('Create iterator.')
    dataloader_iterator = iter(train_dataloader)

    val_err = {}
    logger.info('Start training.')

    try:
        # for step in range(start_iter, max_iters):
        # keep same step in all processes, avoid stuck during eval barrier
        step = start_iter 
        while step < max_iters:
            if main_process():
                training_stats.IterTic()
            
            # get the data batch
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)
            except Exception as e:
                logger.info('When load training data: ', e)
                continue
            except:
                logger.info('Some training data errors exist in the current iter!')
                continue
            data = to_cuda(data)
            # set random crop size
            # if step % 10 == 0:
            #     set_random_crop_size_for_iter(train_dataloader, step, size_sample_list[step])
            
            # check training data
            #for i in range(data['target'].shape[0]):
                # if 'DDAD' in data['dataset'][i] or \
                #     'Lyft' in data['dataset'][i] or \
                #     'DSEC' in data['dataset'][i] or \
                #     'Argovers2' in data['dataset'][i]:
                #     replace = True
                # else:
                #     replace = False
                #visual_train_data(data['target'][i, ...], data['input'][i,...], data['filename'][i], cfg.work_dir, replace=replace)

            # forward
            pred_depth, losses_dict, conf = model(data)
                
            optimizer.zero_grad()
            losses_dict['total_loss'].backward()
            # if step > 100 and step % 10 == 0:
            #     for param in model.parameters():
            #         print(param.grad.max(), torch.norm(param.grad))
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            # Print the profiling results
            print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=10))

            # Optionally, export the profiling results to a file for further analysis
            prof.export_chrome_trace("clip_grad_norm_profile.json")

            optimizer.step()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(losses_dict)

            lr_scheduler.after_train_iter(optimizer)
            if main_process():
                training_stats.update_iter_stats(loss_dict_reduced)
                training_stats.IterToc()
                training_stats.log_iter_stats(step, optimizer, max_iters, val_err)

            # validate the model
            if cfg.evaluation.online_eval and \
                (step+1) % eval_interval == 0 and \
                val_dataloader is not None:
                if isinstance(val_dataloader, list):
                    val_err = validate_multiple_dataset(cfg, step+1, model, val_dataloader, tb_logger)
                else:
                    val_err = validate(cfg, step+1, model, val_dataloader, tb_logger)
                if main_process():
                    training_stats.tb_log_stats(val_err, step)

            # save checkpoint
            if main_process():
                if ((step+1) % save_interval == 0) or ((step+1)==max_iters):
                    save_ckpt(cfg, model, optimizer, lr_scheduler, step+1, epoch)
            
            step += 1

    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)

import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional, cast

import torch
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
import time

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]
def clip_grad_norm2_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> Dict[str, Union[torch.Tensor, float]]:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        A dictionary containing the total norm of the parameter gradients and the durations of each segment.
    """
    durations = {}

    start_time = time.time()
    torch.cuda.synchronize() 
    durations['sync_gpu'] = time.time() - start_time

    start_time = time.time()
    torch.cuda.synchronize() 
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    torch.cuda.synchronize() 
    durations['gradient_collection'] = time.time() - start_time

    # Convert max_norm and norm_type to float
    start_time = time.time()
    torch.cuda.synchronize() 
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return {'total_norm': torch.tensor(0.), 'durations': durations}
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]
    torch.cuda.synchronize() 
    durations['grouping_gradients'] = time.time() - start_time

    # Calculate the norm of the gradients
    start_time = time.time()
    torch.cuda.synchronize() 
    if norm_type == inf:
        norms = [torch.linalg.vector_norm(g.detach(), inf).to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), ([grads], _)) in grouped_grads.items():  # type: ignore[assignment]
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    torch.cuda.synchronize() 
    durations['norm_calculation'] = time.time() - start_time

    # Check for non-finite norms
    start_time = time.time()
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    durations['error_check'] = time.time() - start_time

    # Calculate the clipping coefficient
    start_time = time.time()
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    durations['clipping_coefficient'] = time.time() - start_time

    # Scale the gradients
    start_time = time.time()
    for ((device, _), ([grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):  # type: ignore[arg-type]
            torch._foreach_mul_(grads, clip_coef_clamped.to(device))  # type: ignore[call-overload]
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.detach().mul_(clip_coef_clamped_device)
    durations['gradient_scaling'] = time.time() - start_time

    return {'total_norm': total_norm, 'durations': durations}

def train_by_iters_amp(cfg, model, optimizer, lr_scheduler, train_dataloader, val_dataloader, loss_scaler):
    """
    Do the training by iterations.
    Mix precision is employed.
    """
    # set up logger
    tb_logger = None
    if cfg.use_tensorboard and main_process():
        tb_logger = SummaryWriter(cfg.tensorboard_dir)
    logger = logging.getLogger()
    # training status
    if main_process():
        training_stats = TrainingStats(log_period=cfg.log_interval, tensorboard_logger=tb_logger)

    # learning schedule
    lr_scheduler.before_run(optimizer)
    
    # set training steps
    max_iters = cfg.runner.max_iters
    start_iter = lr_scheduler._step_count

    save_interval = cfg.checkpoint_config.interval
    eval_interval = cfg.evaluation.interval
    epoch = 0

    # If it's too slow try lowering num_worker
    # see https://discuss.pytorch.org/t/define-iterator-on-dataloader-is-very-slow/52238
    logger.info('Create iterator.')
    dataloader_iterator = iter(train_dataloader)

    val_err = {}
    # torch.cuda.empty_cache()
    logger.info('Start training.')

    try:
        acc_batch = cfg.acc_batch
    except:
        acc_batch = 1

    try:
        # for step in range(start_iter, max_iters):
        # keep same step in all processes, avoid stuck during eval barrier
        step = start_iter *  acc_batch
        while step < max_iters:
        # while True:
            durations = {}
            import time
            if main_process():
                training_stats.IterTic()

            # get the data batch
            iteration_start_time = time.time()
            code_start_time = time.time()
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)
            except Exception as e:
                logger.info('When load training data: ', e)
                continue
            except:
                logger.info('Some training data errors exist in the current iter!')
                continue
            durations['data_loading'] = (time.time() - code_start_time)

            code_start_time = time.time()
            data = to_cuda(data)
            durations['to_cuda'] = (time.time() - code_start_time)

            code_start_time = time.time()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred_depth, losses_dict, conf = model(data)
            
            total_loss = losses_dict['total_loss'] / acc_batch

            if not math.isfinite(total_loss):
                logger.info("Loss is {}, skiping this batch training".format(total_loss))
                continue
            durations['forward'] = (time.time() - code_start_time)
            # optimize, backward
            code_start_time = time.time()
            if (step+1-start_iter) % acc_batch == 0:
                optimizer.zero_grad()
                durations['zero_grad'] = (time.time() - code_start_time)
            if loss_scaler == None:
                code_start_time = time.time()
                total_loss.backward()
                durations['backward'] = (time.time() - code_start_time)
                code_start_time = time.time()
                try:
                    if (step+1-start_iter) % acc_batch == 0:
                      
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
                        result_grad_clip = clip_grad_norm2_(model.parameters(), 2.5, norm_type=2.0, error_if_nonfinite=True, foreach=True)
                        durations['clip_grad'] = (time.time() - code_start_time)
                        code_start_time = time.time()
                        optimizer.step()
                        durations['optimizer'] = (time.time() - code_start_time)
                except:
                    print('NAN gradient, skipping optimizer.step() for this round...')
            else:
                loss_scaler(total_loss, optimizer, clip_grad=5, parameters=model.parameters(), update_grad=True)

            # reduce losses over all GPUs for logging purposes
            if (step+1-start_iter) % acc_batch == 0:
                code_start_time = time.time()
                loss_dict_reduced = reduce_dict(losses_dict)
                lr_scheduler.after_train_iter(optimizer)
                durations['lr_schedule'] = (time.time() - code_start_time)
                if main_process():
                    durations['iteration'] = (time.time() - iteration_start_time)
                    training_stats.update_iter_stats(loss_dict_reduced)
                    training_stats.IterToc()
                    training_stats.log_iter_stats(step//acc_batch, optimizer, max_iters, val_err)
                    if (step % 10 == 0):
                        print(
                            f"Step {step//acc_batch+1}/{max_iters} | "
                            f"iteration: {durations.get('iteration', 0):.2f}s | "
                            f"Data loading: {durations.get('data_loading', 0):.5f}s | "
                            f"To CUDA: {durations.get('to_cuda', 0):.5f}s | "
                            f"Forward: {durations.get('forward', 0):.2f}s | "
                            f"zero_grad: {durations.get('zero_grad', 0):.2f}s | "
                            f"Backward: {durations.get('backward', 0):.2f}s | "
                            f"lr_schedule: {durations.get('lr_schedule', 0):.5f}s | "
                            f"clip_grad: {durations.get('clip_grad', 0):.5f}s | "
                            f"Optimizer: {durations.get('optimizer', 0):.2f}s"
                        )

                        if result_grad_clip is not None:
                            print(
                                f"  Grad Clip: total_norm={result_grad_clip.get('total_norm', 'N/A'):.4f} | "
                                + " | ".join([f"{k}: {v:.5f}s" for k, v in result_grad_clip.get('durations', {}).items()])
                            )

            # validate the model
                if cfg.evaluation.online_eval and \
                    ((((step+acc_batch)//acc_batch) % eval_interval == 0) or step == 0) and \
                    val_dataloader is not None:
                # if True:
                    if isinstance(val_dataloader, list):
                        val_err = validate_multiple_dataset(cfg, ((step+acc_batch)//acc_batch), model, val_dataloader, tb_logger)
                    else:
                        val_err = validate(cfg, ((step+acc_batch)//acc_batch), model, val_dataloader, tb_logger)
                    if main_process():
                        training_stats.tb_log_stats(val_err, step)

                # save checkpoint
                if main_process():
                    if (((step+acc_batch)//acc_batch) % save_interval == 0) or (((step+acc_batch)//acc_batch)==max_iters):
                        save_ckpt(cfg, model, optimizer, lr_scheduler, ((step+acc_batch)//acc_batch), epoch, loss_scaler=loss_scaler)

            step += 1
            

    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)

def validate_multiple_dataset(cfg, iter, model, val_dataloaders, tb_logger):
    val_errs = {}
    for val_dataloader in val_dataloaders:
        val_err = validate(cfg, iter, model, val_dataloader, tb_logger)
        val_errs.update(val_err)
    # mean of all dataset
    mean_val_err = {}
    for k, v in val_errs.items():
        metric = 'AllData_eval/' + k.split('/')[-1]
        if metric not in mean_val_err.keys():
            mean_val_err[metric] = 0
        mean_val_err[metric] += v / len(val_dataloaders)
    val_errs.update(mean_val_err)
    
    return val_errs


def validate(cfg, iter, model, val_dataloader, tb_logger):
    """
    Validate the model on single dataset
    """
    model.eval()
    if cfg.distributed:
        dist.barrier()
    logger = logging.getLogger()
    # prepare dir for visualization data
    save_val_meta_data_dir = create_dir_for_validate_meta(cfg.work_dir, iter)
    # save_html_path = save_val_meta_data_dir + '.html'
    # dataset_name = val_dataloader.dataset.data_name
    from torch.utils.data import ConcatDataset
    if isinstance(val_dataloader.dataset, ConcatDataset):
        dataset_names = [d.data_name for d in val_dataloader.dataset.datasets if hasattr(d, 'data_name')]
        dataset_name = ','.join(dataset_names)  # Combine names if needed
    else:
        dataset_name = val_dataloader.dataset.data_name

    # save_point = max(int(len(val_dataloader) / 5), 1)
    # save_point = 2
    # depth metric meter
    dam = MetricAverageMeter(cfg.evaluation.metrics)
    # dam_disp = MetricAverageMeter([m for m in cfg.evaluation.metrics if m[:6]!='normal'])
    for i, data in enumerate(val_dataloader):
        if i % 10 == 0:
            logger.info(f'Validation step on {dataset_name}: {i}')
        data = to_cuda(data)
        output = model.module.inference(data)
        pred_depth = output['prediction']
        pred_depth = pred_depth.squeeze()
        gt_depth = data['target'].cuda(non_blocking=True).squeeze()
        
        pad = data['pad'].squeeze()
        H, W = pred_depth.shape
        pred_depth = pred_depth[pad[0]:H-pad[1], pad[2]:W-pad[3]]
        gt_depth = gt_depth[pad[0]:H-pad[1], pad[2]:W-pad[3]]
        rgb = data['input'][0, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        mask = gt_depth > 0
        #pred_depth_resize = cv2.resize(pred_depth.cpu().numpy(), (torch.squeeze(data['B_raw']).shape[1], torch.squeeze(data['B_raw']).shape[0]))
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, cfg.distributed)

        # save evaluation results
        save_point = max(int(len(val_dataloader) / 5), 1)
        if i%save_point == 0 and main_process():
            save_val_imgs(iter, 
                          pred_depth, 
                          gt_depth, 
                          rgb, # data['input'], 
                          dataset_name + '_' + data['filename'][0], 
                          save_val_meta_data_dir,
                          tb_logger=tb_logger)

        ## surface normal
        if "normal_out_list" in output.keys():
            normal_out_list = output['normal_out_list']
            pred_normal = normal_out_list[-1][:, :3, :, :] # (B, 3, H, W)
            gt_normal = data['normal'].cuda(non_blocking=True)
            # if pred_normal.shape != gt_normal.shape:
            #     pred_normal = F.interpolate(pred_normal, size=[gt_normal.size(2), gt_normal.size(3)], mode='bilinear', align_corners=True)

            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            gt_normal = gt_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            gt_normal_mask = ~torch.all(gt_normal == 0, dim=1, keepdim=True)
            dam.update_normal_metrics_gpu(pred_normal, gt_normal, gt_normal_mask, cfg.distributed)

            # save valiad normal
            if i%save_point == 0 and main_process():
                save_normal_val_imgs(iter, 
                                    pred_normal, 
                                    gt_normal, 
                                    rgb, # data['input'], 
                                    dataset_name + '_normal_' + data['filename'][0], 
                                    save_val_meta_data_dir,
                                    tb_logger=tb_logger)

    # create html for visualization
    merged_rgb_pred_gt = os.path.join(save_val_meta_data_dir, '*_merge.jpg')
    name2path = dict(merg=merged_rgb_pred_gt) #dict(rgbs=rgbs, pred=pred, gt=gt)
    # if main_process():
    #    create_html(name2path, save_path=save_html_path, size=(256*3, 512))

    # get validation error
    eval_error = dam.get_metrics()
    eval_error = {f'{dataset_name}_eval/{k}': v for k,v in eval_error.items()}
    # eval_disp_error = {f'{dataset_name}_eval/disp_{k}': v for k,v in dam_disp.get_metrics().items()}
    # eval_error.update(eval_disp_error)

    model.train()
    
    if 'exclude' in cfg.evaluation and dataset_name in cfg.evaluation.exclude:
        return {}
    return eval_error

def set_random_crop_size_for_iter(dataloader: torch.utils.data.dataloader.DataLoader, iter: int, size_pool=None):
    if size_pool is None:
        size_pool = [
            # [504, 504], [560, 1008], [840, 1512], [1120, 2016],
            [560, 1008], [840, 1512], [1120, 2016],
            # [480, 768], [480, 960], 
            # [480, 992], [480, 1024], 
            # [480, 1120], 
            # [480, 1280], 
            # [480, 1312],
            # [512, 512], [512, 640], 
            # [512, 960], 
            # [512, 992], 
            # [512, 1024], [512, 1120], 
            # [512, 1216], 
            # [512, 1280],
            # [576, 640], [576, 960], 
            # [576, 992], 
            # [576, 1024],
            # [608, 608], [608, 640], 
            # [608, 960], [608, 1024],
        ]
    random.seed(iter)
    sample = random.choice(size_pool)
    # idx = (iter // 10) % len(size_pool)
    #sample = size_pool[size_idx]
    
    # random.seed(iter)
    # flg = random.random() <= 1.0
    # if flg:
    crop_size = sample
    # else:
    #     crop_size = [sample[1], sample[0]]

    # set crop size for each dataset
    datasets_groups = len(dataloader.dataset.datasets)
    for i in range(datasets_groups):
        for j in range(len(dataloader.dataset.datasets[i].datasets)):
            dataloader.dataset.datasets[i].datasets[j].set_random_crop_size(crop_size)
    return crop_size


