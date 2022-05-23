import argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import config
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset
from src.utils.Renderer import Renderer
from src.NICE_SLAM import NICE_SLAM
from src.common import get_camera_from_tensor
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    device = 'cuda:1'
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')  # args.config: env, nice_slam.yaml: robot
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output

    slam = NICE_SLAM(cfg, args)
    decoders = torch.load('decoders.pth')
    renderer = slam.renderer

    frame_reader = get_dataset(cfg, args, scale, device=device)
    frame_loader = DataLoader(
            frame_reader, batch_size=1, shuffle=False, num_workers=4)
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device(device))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
    estimate_c2w_list[:, :3, 3] /= scale
    gt_c2w_list[:, :3, 3] /= scale

    c = np.load('c.npy',allow_pickle=True).item()
    
    for i in tqdm(range(0, N+1)):
        print(N)
        idx, gt_color, gt_depth, gt_c2w = frame_reader[i]
        print(gt_c2w)
        c2w = torch.tensor([[-3.1540e-01,  2.8231e-01,  2.5936e-01, -1.5962e+00],
        [ 8.7653e-01,  5.0214e-02,  3.1161e-01, -3.3759e-01],
        [ 2.2031e-16,  7.8242e-01, -1.8669e-01,  2.0815e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).to(device)
        
        depth, _, _ = renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    device,
                    stage='color',
                    gt_depth=None)
        depth, uncertainty, color = renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    device,
                    stage='color',
                    gt_depth=depth)
        gt_depth_np = gt_depth.cpu()
        gt_color_np = gt_color.cpu()
        depth_np = depth.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
        #print(color_np)

        axs[0, 0].imshow(gt_depth_np, cmap='plasma')
        axs[0, 0].set_title('Ground Truth Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap='plasma')
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        axs[1, 0].imshow(gt_color_np)
        axs[1, 0].set_title('Ground Truth RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np)
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        plt.show()
        plt.savefig('test.png')#, bbox_inches='tight', pad_inches=0.2)
