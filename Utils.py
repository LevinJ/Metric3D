# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys, time,torch,torchvision,pickle,trimesh,itertools,datetime,imageio,logging,joblib,importlib,argparse
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import pandas as pd
import open3d as o3d
import cv2
import numpy as np
from transformations import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)



def set_logging_format(level=logging.INFO):
  importlib.reload(logging)
  FORMAT = '%(message)s'
  logging.basicConfig(level=level, format=FORMAT, datefmt='%m-%d|%H:%M:%S')

set_logging_format()



def set_seed(random_seed):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud



def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
  invalid_mask = (depth<zmin)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0
  return xyz_map



def freeze_model(model):
  model = model.eval()
  for p in model.parameters():
    p.requires_grad = False
  for p in model.buffers():
    p.requires_grad = False
  return model



def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H%divider==0
  assert max_W%divider==0

  def round_by_divider(x):
    return int(np.ceil(x/divider)*divider)

  H_resize = round_by_divider(H)   #!NOTE KITTI width=1242
  W_resize = round_by_divider(W)
  if H_resize>max_H or W_resize>max_W:
    if H_resize>W_resize:
      W_resize = round_by_divider(W_resize*max_H/H_resize)
      H_resize = max_H
    else:
      H_resize = round_by_divider(H_resize*max_W/W_resize)
      W_resize = max_W
  return int(H_resize), int(W_resize)


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)



def depth_uint8_decoding(depth_uint8, scale=1000):
  depth_uint8 = depth_uint8.astype(float)
  out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
  return out/float(scale)


def process_and_visualize_point_cloud(
    depth: np.ndarray,  # (H, W) - Depth map
    K: np.ndarray,      # (3, 3) - Camera intrinsic matrix
    img: np.ndarray,    # (H, W, 3) - RGB image
    out_dir: str,       # Path to output directory
    z_far: float,       # Maximum depth to clip in the point cloud
    denoise_cloud: bool = False,  # Whether to denoise the point cloud
    denoise_nb_points: int = 30,  # Number of points for radius outlier removal
    denoise_radius: float = 0.03  # Radius for outlier removal
) -> None:
    """
    Generates, filters, and visualizes a point cloud from depth data.

    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        K (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        img (np.ndarray): Original RGB image of shape (H, W, 3) for coloring the point cloud.
        out_dir (str): Directory to save the point cloud files.
        z_far (float): Maximum depth to clip in the point cloud.
        denoise_cloud (bool): Whether to denoise the point cloud.
        denoise_nb_points (int): Number of points to consider for radius outlier removal.
        denoise_radius (float): Radius to use for outlier removal.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Generate the point cloud
    xyz_map = depth2xyzmap(depth, K)  # (H, W, 3)
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img.reshape(-1, 3))  # (N, 3)

    # Filter the point cloud based on depth
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {out_dir}")

    # Optionally denoise the point cloud
    if denoise_cloud:
        logging.info("Denoising point cloud...")
        _, ind = pcd.remove_radius_outlier(nb_points=denoise_nb_points, radius=denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(f'{out_dir}/cloud_denoise.ply', inlier_cloud)
        pcd = inlier_cloud

    # Visualize the point cloud
    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()

