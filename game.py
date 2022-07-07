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
from scipy.spatial.transform import Rotation as R

device = 'cuda:1'
time_step = 0.1
mu = 0
sigma = 5
alpha = 0.5
d = 0.1

def state_to_pose(state):
    for i in range(state.shape[0]):
        rm = torch.from_numpy(R.from_euler('xyz', state[i][3:].cpu(), degrees=True).as_matrix()).to(device)
        pose = torch.cat((torch.cat((rm, state[i][:3].unsqueeze(-1)), 1), torch.tensor([[0,0,0,1]]).to(device)), 0).unsqueeze(0)
        if i == 0:
            batch_pose = pose
        else:
            batch_pose = torch.cat((batch_pose, pose), dim=0)
    return batch_pose

def update_dynamics(state, action):
    return state + time_step * action


class Robot:
    def __init__(self, cfg):
        self.c = np.load('c.npy',allow_pickle=True).item()
        self.decoders = torch.load('decoders.pth')
        slam = NICE_SLAM(cfg, args)
        self.renderer = slam.renderer
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy

    def predict_observation(self, pose):
        depth, _, _ = self.renderer.render_batch_img(
                    self.c,
                    self.decoders,
                    pose,
                    device,
                    stage='middle',
                    gt_depth=None)
        return depth
    
    def render(self, pose):
        depth, _, _ = self.renderer.render_img(
                    self.c,
                    self.decoders,
                    pose,
                    device,
                    stage='color',
                    gt_depth=None)
        depth, uncertainty, color = self.renderer.render_img(
                    self.c,
                    self.decoders,
                    pose,
                    device,
                    stage='color',
                    gt_depth=depth)
        return depth, color



def find_safe_action(robot, pose, h, intended_action, direction):
    state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0)
    orient_action = torch.zeros(6).to(device)
    if direction in ['up', 'down']:
        orient_action[0] = intended_action * pose[0][0]
        orient_action[1] = intended_action * pose[0][1]
        orient_action[2] = intended_action * pose[0][2]
    elif direction in ['left', 'right']:
        orient_action[5] = intended_action * 50
    new_state = update_dynamics(state, orient_action).unsqueeze(0)
    new_pose = state_to_pose(new_state)
    new_h = 0.1 - robot.predict_observation(new_pose).min().unsqueeze(0)
    best_action = torch.zeros(6)
    if new_h <= alpha * h:
        print('Intended action {} is safe'.format(orient_action))
        return orient_action
    for i in range(10):
        batch_action = torch.zeros((10, 6)).to(device)
        if direction == 'up':
            for j in range(10):
                value = abs(np.random.normal(mu, sigma))
                for k in range(3):
                    batch_action[j][k] = value * pose[k][0]
        elif direction == 'down':
            for j in range(10):
                value = -abs(np.random.normal(mu, sigma))
                for k in range(3):
                    batch_action[j][k] = value * pose[k][0]
        elif direction == 'left':
            for j in range(10):
                batch_action[j][5] = abs(np.random.normal(mu, sigma))
        elif direction == 'right':
            for j in range(10):
                batch_action[j][5] = -abs(np.random.normal(mu, sigma))
        batch_new_state = update_dynamics(state, batch_action)
        batch_new_pose = state_to_pose(batch_new_state)
        batch_new_h = 0.1 - robot.predict_observation(batch_new_pose).min(dim=-1)[0].min(dim=-1)[0]
        for j in range(10):
            if batch_new_h[j] <= alpha * h:
                if abs(sum(batch_action[j] ** 2) - intended_action ** 2) < abs(sum(best_action ** 2) - intended_action ** 2):
                    best_action = batch_action[j]
        if sum(best_action ** 2) > 0:
            print('Intended action {} is unsafe, a recommended substitute is {}'.format(orient_action, best_action))
            return best_action
    print('Fail to find a safe action')
    return best_action


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
    
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')  # args.config: env, nice_slam.yaml: robot
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output

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

    robot = Robot(cfg)
    
    idx, color, depth, pose = frame_reader[0]
    
    depth, color = robot.render(pose.to(device))
    cv2.namedWindow("Safety Filter", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
    dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))

    while True:
        k = cv2.waitKeyEx()
        start = time.time()
        if k == 65362:  # up
            intended_action = abs(np.random.normal(mu, sigma))
            action = find_safe_action(robot, pose, d - depth.min(), intended_action, 'up')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            old_state = state
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))
            end = time.time()
            print('old state = {}, new state = {}, min_depth = {}, time cost = {}'.format(old_state, state, depth.min(), end-start))
            print('new pose = {}'.format(pose))
        elif k == 65364:  # down
            intended_action = -abs(np.random.normal(mu, sigma))
            action = find_safe_action(robot, pose, d - depth.min(), intended_action, 'down')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            old_state = state
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))
            end = time.time()
            print('old state = {}, new state = {}, min_depth = {}, time cost = {}'.format(old_state, state, depth.min(), end-start))
            print('new pose = {}'.format(pose))
        elif k == 65361:  # left
            intended_action = abs(np.random.normal(mu, sigma))
            action = find_safe_action(robot, pose, d - depth.min(), intended_action, 'left')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            old_state = state
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))
            end = time.time()
            print('old state = {}, new state = {}, min_depth = {}, time cost = {}'.format(old_state, state, depth.min(), end-start))
            print('new pose = {}'.format(pose))
        elif k == 65363:  # right
            intended_action = -abs(np.random.normal(mu, sigma))
            action = find_safe_action(robot, pose, d - depth.min(), intended_action, 'right')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            old_state = state
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))
            end = time.time()
            print('old state = {}, new state = {}, min_depth = {}, time cost = {}'.format(old_state, state, depth.min(), end-start))
            print('new pose = {}'.format(pose))
        elif k == 27:  # esc
            cv2.destroyAllWindows()
            break
        
