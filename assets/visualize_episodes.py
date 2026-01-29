#!/usr/bin/env python3
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import cv2

def load_hdf5_data(dataset_path):
    """加载HDF5数据文件"""
    with h5py.File(dataset_path, 'r') as f:
        # 打印数据结构以便调试
        print("\nHDF5 file structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)
        
        # 加载数据
        data = {}
        
        # 读取观测数据
        if 'observations' in f:
            obs = f['observations']
            if 'qpos' in obs:
                data['qpos'] = obs['qpos'][:]
            if 'qvel' in obs:
                data['qvel'] = obs['qvel'][:]
            
            # 读取图像数据
            if 'images' in obs:
                images = {}
                for cam_name in obs['images'].keys():
                    images[cam_name] = obs['images'][cam_name][:]
                data['images'] = images
        
        # 读取动作数据
        if 'action' in f:
            data['action'] = f['action'][:]
        
        # 读取属性
        data['attrs'] = dict(f.attrs)
        
    return data

def create_video_from_images(images_dict, output_path, fps=30):
    """从图像字典创建视频"""
    # 获取第一个相机的图像来确定尺寸
    cam_names = list(images_dict.keys())
    if not cam_names:
        print("No camera images found")
        return
    
    first_cam_images = images_dict[cam_names[0]]
    num_frames = len(first_cam_images)
    
    if num_frames == 0:
        print("No frames to process")
        return
    
    # 如果有多个相机，拼接成一行
    if len(cam_names) > 1:
        # 获取所有相机的图像
        all_cam_images = []
        for cam in cam_names:
            if cam in images_dict:
                all_cam_images.append(images_dict[cam])
        
        # 确保所有相机有相同数量的帧
        min_frames = min(len(imgs) for imgs in all_cam_images)
        
        # 创建拼接的视频
        first_frame = np.hstack([imgs[0] for imgs in all_cam_images])
        height, width = first_frame.shape[:2]
    else:
        all_cam_images = [first_cam_images]
        min_frames = num_frames
        height, width = first_cam_images[0].shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 写入帧
    for i in range(min_frames):
        if len(all_cam_images) > 1:
            frame = np.hstack([imgs[i] for imgs in all_cam_images])
        else:
            frame = all_cam_images[0][i]
        
        # 转换RGB到BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to: {output_path}")

def visualize_joints(qpos, action=None, output_path=None):
    """可视化关节轨迹"""
    num_timesteps, num_joints = qpos.shape
    
    # 动态生成关节名称
    joint_names = [f'joint_{i}' for i in range(num_joints)]
    
    # 为主要关节创建更有意义的名称（根据你的机器人调整）
    if num_joints >= 14:  # 假设有两个6DOF手臂 + 其他关节
        joint_names[:7] = [f'left_arm_j{i+1}' for i in range(7)]
        joint_names[7:14] = [f'right_arm_j{i+1}' for i in range(7)]
        if num_joints > 14:
            joint_names[14] = 'platform_joint'
        if num_joints > 15:
            joint_names[15] = 'head_joint1'
        if num_joints > 16:
            joint_names[16] = 'head_joint2'
    
    print(f"\nVisualizing {num_joints} joints over {num_timesteps} timesteps")
    
    # 选择要可视化的关节（最多显示16个）
    max_plots = min(16, num_joints)
    selected_joints = list(range(max_plots))
    
    # 创建子图
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for plot_idx, joint_idx in enumerate(selected_joints):
        ax = axes[plot_idx]
        
        # 绘制qpos轨迹
        ax.plot(qpos[:, joint_idx], label='qpos', color='blue', linewidth=1.5)
        
        # 如果有action数据，也绘制它
        if action is not None and joint_idx < action.shape[1]:
            ax.plot(action[:, joint_idx], label='action', color='red', 
                   linewidth=1.5, linestyle='--', alpha=0.7)
        
        ax.set_title(f'{joint_names[joint_idx]}', fontsize=10)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # 隐藏未使用的子图
    for idx in range(max_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Joint Trajectories', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved joint plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_statistics(data):
    """打印数据统计信息"""
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    
    if 'qpos' in data:
        qpos = data['qpos']
        print(f"\nqpos:")
        print(f"  Shape: {qpos.shape}")
        print(f"  Range: [{np.min(qpos):.3f}, {np.max(qpos):.3f}]")
        print(f"  Mean: {np.mean(qpos):.3f}, Std: {np.std(qpos):.3f}")
    
    if 'action' in data:
        action = data['action']
        print(f"\naction:")
        print(f"  Shape: {action.shape}")
        print(f"  Range: [{np.min(action):.3f}, {np.max(action):.3f}]")
        print(f"  Mean: {np.mean(action):.3f}, Std: {np.std(action):.3f}")
    
    if 'images' in data:
        print(f"\nImages:")
        for cam_name, imgs in data['images'].items():
            print(f"  {cam_name}: shape={imgs.shape}, dtype={imgs.dtype}")
    
    if 'attrs' in data:
        print(f"\nAttributes:")
        for key, value in data['attrs'].items():
            print(f"  {key}: {value}")

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    
    # 构建文件路径
    hdf5_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    # 检查文件是否存在
    if not os.path.exists(hdf5_path):
        # 尝试其他可能的文件名
        alt_path = os.path.join(dataset_dir, 'episode.hdf5')
        if os.path.exists(alt_path):
            hdf5_path = alt_path
            print(f"Using alternative path: {alt_path}")
        else:
            print(f"Error: File not found: {hdf5_path}")
            return
    
    print(f"Loading data from: {hdf5_path}")
    
    # 加载数据
    data = load_hdf5_data(hdf5_path)
    
    # 打印统计信息
    print_statistics(data)
    
    # 创建视频
    if 'images' in data:
        video_path = os.path.join(dataset_dir, f'episode_{episode_idx}_video.mp4')
        create_video_from_images(data['images'], video_path, fps=30)
    
    # 可视化关节轨迹
    if 'qpos' in data:
        plot_path = os.path.join(dataset_dir, f'episode_{episode_idx}_qpos.png')
        action = data.get('action', None)
        visualize_joints(data['qpos'], action, plot_path)
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--episode_idx', type=int, default=0,
                       help='Episode index to visualize')
    
    main(vars(parser.parse_args()))