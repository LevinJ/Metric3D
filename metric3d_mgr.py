"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Thu Mar 13 2025
*  File : metric3d_mgr.py
******************************************* -->

"""

# Import the Metric class from the appropriate module
from PIL import Image
from typing import Optional
from matplotlib import scale
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
import time  # Ensure this import is at the top of the file

import numpy as np
import onnxruntime as ort
# from scripts.metric import Metric3D  # Adjust the module name as needed

class Metric3dMgr(object):
    def __init__(self):
        self.model = None
        self.use_onnx = True
        return
    def init_model(self, model_name='metric3d_vit_giant2'):
        if self.model is not None:
            return
        if self.use_onnx:
            onnx_model="/media/levin/DATA/checkpoints/droid_metric/metric3d_vit_small.onnx"
            # onnx_model="/media/levin/DATA/checkpoints/droid_metric/metric3d_vit_small.onnx"
            # providers = [
            #     (
            #         "CUDAExecutionProvider",
            #         {"cudnn_conv_use_max_workspace": "0", "device_id": str(0)},
            #     )
            # ]
            providers = [("TensorrtExecutionProvider", {'trt_engine_cache_enable': True, 'trt_fp16_enable': True, 'device_id': 0, 'trt_dla_enable': False})]
            self.model = ort.InferenceSession(onnx_model, providers=providers)
            return
        # Use torch.hub.load to load the model from a local directory
        local_repo = str(Path(__file__).parent)  # Get the current file's directory
        model = torch.hub.load(local_repo, model_name, source='local', pretrain=True)
        model.cuda().eval()
        self.model = model
        return
    # def infer(self,rgb_image, intr):
    #     depth = self.metric(rgb_image=rgb_image, intrinsic=intr, d_max=self.d_max)
    #     return depth

    def prepare_input(self, rgb_origin: np.ndarray, input_size: tuple = (616, 1064)) -> tuple:
        """
        Prepares the input image and intrinsic parameters for the model.

        Args:
            rgb_origin (np.ndarray): Original RGB image.
            intrinsic (list): Intrinsic camera parameters [fx, fy, cx, cy].
            input_size (tuple): Target input size for the model (height, width).

        Returns:
            tuple: Processed RGB tensor and padding information.
        """
        # Keep ratio resize
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Scale intrinsic parameters
        # intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

        # Padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        if self.use_onnx:
            return rgb, pad_info, scale

        # Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        return rgb, pad_info, scale
    def infer_depth(self, rgb_origin: np.ndarray, intrinsic: list) -> np.ndarray:
        #### Call prepare_input
        rgb, pad_info, scale = self.prepare_input(rgb_origin, input_size=(616, 1064))

        ###################### canonical camera space ######################
        start_time = time.time()  # Start timing
        if self.use_onnx:
            onnx_input = {
            "pixel_values": np.ascontiguousarray(
                np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32
            ),  # 1, 3, H, W
    }
            outputs = self.model.run(None, onnx_input)
            pred_depth = outputs[0].squeeze()  # [H, W]
            pred_depth = torch.from_numpy(pred_depth).cuda()
        else:
            with torch.no_grad():
                pred_depth, confidence, output_dict = self.model.inference({'input': rgb})
        print(f"Model inference duration: {time.time() - start_time:.3f} seconds")  # Print the duration with 3-digit precision

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
        ###################### canonical camera space ######################

        #### de-canonical transform
        canonical_to_real_scale = (intrinsic[0] + intrinsic[1]) * scale / (2 * 1000.0) # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)
        return pred_depth

    def run(self):
        self.init_model( model_name='metric3d_vit_small')
        #### prepare data
        rgb_file = '/media/levin/DATA/zf/nerf/2024_0601/scenes/5/rgb/es81_sur_back/rgb_00655_sur_back.jpg'
        depth_file = 'small_torch.npy'
        intrinsic = [1081.695079, 1081.019193, 950.014133, 557.173103]
        gt_depth_scale = 256.0
        rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

       
        pred_depth = self.infer_depth(rgb_origin, intrinsic)
        # save predicted depth
        # np.save('small_torch.npy', pred_depth.cpu().numpy())

        #### you can now do anything with the metric depth 
        # such as evaluate predicted depth
        if depth_file is not None:
            if depth_file.endswith('.npy'):
                gt_depth = np.load(depth_file)
                gt_depth_scale = 1.0
            else:
                gt_depth = cv2.imread(depth_file, -1)
            gt_depth = gt_depth / gt_depth_scale
            gt_depth = torch.from_numpy(gt_depth).float().cuda()
            assert gt_depth.shape == pred_depth.shape
            
            mask = (gt_depth > 1e-8)
            abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
            print('abs_rel_err:', abs_rel_err.item())

        #### normal are also available
        # if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
        #     pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        #     normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
        #     # un pad and resize to some size if needed
        #     pred_normal = pred_normal.squeeze()
        #     pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
        #     # you can now do anything with the normal
        #     # such as visualize pred_normal
        #     pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        #     pred_normal_vis = (pred_normal_vis + 1) / 2
        #     cv2.imwrite('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))

        return 

if __name__ == "__main__":   
    obj = Metric3dMgr()
    obj.run()
