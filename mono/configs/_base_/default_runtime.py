# distributed training configs, if  dist_url == 'env://'('tcp://127.0.0.1:6795'), nodes related configs should be set in the shell
dist_params = dict(port=None, backend='nccl', dist_url='env://', nnodes=1, node_rank=0)

load_from = None
cudnn_benchmark = True
test_metrics = ['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3','rmse_log', 'log10', 'sq_rel']
