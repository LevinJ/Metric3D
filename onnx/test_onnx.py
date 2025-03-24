import torch
import onnxruntime as ort
import numpy as np
import cv2
from typing import Tuple, Dict, List
from matplotlib import pyplot as plt
import time  # Add this import



def prepare_input(
    rgb_image: np.ndarray, input_size: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], List[int]]:

    h, w = rgb_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(
        rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )

    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb: np.ndarray = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    onnx_input = {
        "image": np.ascontiguousarray(
            np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32
        ),  # 1, 3, H, W
    }
    return onnx_input, pad_info, scale


def main(
    onnx_model="metric3d_vit_small.onnx",
    input_image="data/kitti_demo/rgb/0000000100.png",
):
    intrinsic = [1081.695079, 1081.019193, 950.014133, 557.173103]
    input_image = '/media/levin/DATA/zf/nerf/2024_0601/scenes/5/rgb/es81_sur_back/rgb_00655_sur_back.jpg'
    depth_file = 'small_torch.npy'
    ## Dummy Test
    B = 1
    if "vit" in onnx_model:
        input_size = (616, 1064)  # [H, W]
        dummy_image = np.zeros([B, 3, input_size[0], input_size[1]], dtype=np.float32)
    else:
        input_size = (544, 1216)  # [H, W]
        dummy_image = np.zeros([B, 3, input_size[0], input_size[1]], dtype=np.float32)

    providers = [
        (
            "CUDAExecutionProvider",
            {"cudnn_conv_use_max_workspace": "0", "device_id": str(0)},
        )
    ]
    # providers = [("TensorrtExecutionProvider", {'trt_engine_cache_enable': True, 'trt_fp16_enable': True, 'device_id': 0, 'trt_dla_enable': False})]
    ort_session = ort.InferenceSession(onnx_model, providers=providers)
    
    start_time = time.time()  # Start timing
    outputs = ort_session.run(None, {"image": dummy_image})
    end_time = time.time()  # End timing

    print(f"Runtime duration for ort_session.run: {end_time - start_time:.4f} seconds")

    print(
        f"The actual output of onnxruntime session for the dummy set: outputs[0].shape={outputs[0].shape}"
    )

    ## Real Test
    rgb_image = cv2.imread(input_image)[:, :, ::-1]  # BGR to RGB
    original_shape = rgb_image.shape[:2]
    

    start_time = time.time()  # Start timing
    onnx_input, pad_info, scale = prepare_input(rgb_image, input_size)
    outputs = ort_session.run(None, onnx_input)
    end_time = time.time()  # End timing

    print(f"Runtime duration for prepare_input and ort_session.run: {end_time - start_time:.4f} seconds")
    
    depth = outputs[0].squeeze()  # [H, W]

    # Reshape the depth to the original size
    depth = depth[
        pad_info[0] : input_size[0] - pad_info[1],
        pad_info[2] : input_size[1] - pad_info[3],
    ]
    depth = cv2.resize(
        depth, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR
    )

    canonical_to_real_scale = (intrinsic[0] + intrinsic[1]) * scale / (2 * 1000.0) # 1000.0 is the focal length of canonical camera
    depth = depth * canonical_to_real_scale # now the depth is metric
    # np.save("small_onnx.npy", depth)
    # such as evaluate predicted depth
    if depth_file is not None:
        gt_depth_scale = 256.0
        pred_depth = torch.from_numpy(depth).cuda()
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


    plt.subplot(1, 2, 1)
    plt.imshow(depth)
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_image)
    plt.show()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    print("Done!")
