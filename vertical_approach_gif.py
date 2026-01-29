#!/usr/bin/env python3
"""演示左侧末端执行器移动到方块上方并调整为垂直抓取姿态，生成GIF动画"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ee_sim_env import make_ee_sim_env
from scipy.spatial.transform import Rotation as R
import os

def create_vertical_approach_gif():
    """创建左手垂直抓取姿态演示的GIF动画"""
    print("=== 创建左手垂直抓取姿态GIF动画 ===")
    
    # 创建环境
    env = make_ee_sim_env('sim_transfer_cube_scripted')
    ts = env.reset()
    
    # 获取方块位置
    cube_pos = ts.observation.get('object', np.array([0.3, -0.6, 0.6]))
    print(f"方块位置: {cube_pos}")
    
    # 获取初始机械臂位置
    initial_left_pos = ts.observation.get('mocap_pose_left', np.array([0.4, -0.6, 0.65]))[:3]
    initial_right_pos = ts.observation.get('mocap_pose_right', np.array([-0.4, -0.6, 0.65]))[:3]
    
    print(f"初始左臂位置: {initial_left_pos}")
    print(f"初始右臂位置: {initial_right_pos}")
    
    # 收集所有帧用于GIF
    gif_frames = []
    frame_count = 0
    
    def add_frame(ts, label):
        nonlocal frame_count
        if 'images' in ts.observation and 'top' in ts.observation['images']:
            frame = ts.observation['images']['top'].copy()
            gif_frames.append(frame)
            frame_count += 1
            print(f"Frame {frame_count}: {label}")
    
    # 添加初始状态（停留1秒 = 20帧）
    for i in range(20):
        add_frame(ts, f"初始状态 {i+1}/20")
    
    # 计算目标位置（方块上方10cm）
    target_pos = cube_pos.copy()
    target_pos[2] += 0.10  # 上方10cm
    print(f"目标位置（方块上方10cm）: {target_pos}")
    
    # 定义垂直向下的姿态
    # 指尖垂直于桌面向下 - 绕X轴旋转90度
    vertical_quat = np.array([0.707, 0.707, 0.0, 0.0])  # [w, x, y, z]
    print(f"垂直姿态四元数: {vertical_quat}")
    
    print("\n=== 开始生成动画帧 ===")
    
    # 阶段1: 平滑移动到目标位置并调整姿态（3秒 = 60帧）
    print("\n阶段1: 移动到方块上方10cm并调整为垂直姿态")
    
    steps_phase1 = 60
    start_pos = initial_left_pos.copy()
    start_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 初始水平姿态
    
    for step in range(steps_phase1):
        # 平滑插值参数（使用缓动函数）
        t_raw = (step + 1) / steps_phase1
        # 使用缓入缓出曲线
        t = 3 * t_raw * t_raw - 2 * t_raw * t_raw * t_raw
        
        # 位置插值
        current_pos = start_pos * (1 - t) + target_pos * t
        
        # 姿态球面线性插值
        try:
            start_rot = R.from_quat([start_quat[1], start_quat[2], start_quat[3], start_quat[0]])
            target_rot = R.from_quat([vertical_quat[1], vertical_quat[2], vertical_quat[3], vertical_quat[0]])
            
            # 球面线性插值
            current_rot = start_rot.slerp(target_rot, t)
            current_quat_scipy = current_rot.as_quat()
            current_quat = np.array([current_quat_scipy[3], current_quat_scipy[0], 
                                   current_quat_scipy[1], current_quat_scipy[2]])
        except:
            # 线性插值后归一化
            current_quat = start_quat * (1 - t) + vertical_quat * t
            current_quat = current_quat / np.linalg.norm(current_quat)
        
        # 构造动作
        # 左手：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, gripper]
        action_left = np.concatenate([current_pos, current_quat, [1.0]])  # 夹爪张开
        
        # 右手保持不动
        action_right = np.concatenate([initial_right_pos, [1,0,0,0], [0.5]])
        
        # 合并动作
        action = np.concatenate([action_left, action_right])
        
        # 执行动作
        ts = env.step(action)
        
        # 添加到GIF帧
        add_frame(ts, f"移动中 {step+1}/{steps_phase1}")
    
    # 阶段2: 稳定保持姿态（1秒 = 20帧）
    print("\n阶段2: 稳定保持垂直姿态")
    
    final_action_left = np.concatenate([target_pos, vertical_quat, [1.0]])
    final_action_right = np.concatenate([initial_right_pos, [1,0,0,0], [0.5]])
    final_action = np.concatenate([final_action_left, final_action_right])
    
    for step in range(20):
        ts = env.step(final_action)
        add_frame(ts, f"稳定中 {step+1}/20")
    
    # 阶段3: 最终停留（1秒 = 20帧）
    print("\n阶段3: 最终垂直姿态展示")
    for step in range(20):
        ts = env.step(final_action)
        add_frame(ts, f"最终展示 {step+1}/20")
    
    print(f"\n✅ 共收集 {len(gif_frames)} 帧图像")
    
    # 创建GIF动画
    if gif_frames:
        print("正在创建GIF动画...")
        
        # 转换numpy数组为PIL图像
        pil_frames = []
        for frame in gif_frames:
            # 确保是uint8格式
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(frame)
            pil_frames.append(pil_image)
        
        # 保存GIF（帧率约20fps，每帧50ms）
        gif_path = 'vertical_approach_animation.gif'
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=50,  # 每帧50ms
            loop=0  # 无限循环
        )
        
        print(f"✅ GIF动画已保存: {gif_path}")
        print(f"   总帧数: {len(pil_frames)}")
        print(f"   动画时长: {len(pil_frames) * 0.05:.1f}秒")
        print(f"   文件大小: {os.path.getsize(gif_path) / 1024 / 1024:.2f}MB")
        
        # 创建静态预览图像
        preview_frames = [0, len(pil_frames)//4, len(pil_frames)//2, 
                         3*len(pil_frames)//4, len(pil_frames)-1]
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        stage_names = ["初始状态", "开始移动", "移动中", "接近目标", "最终姿态"]
        
        for i, (frame_idx, stage_name) in enumerate(zip(preview_frames, stage_names)):
            axes[i].imshow(gif_frames[frame_idx])
            axes[i].set_title(f"{stage_name}\n(帧 {frame_idx+1}/{len(gif_frames)})", fontsize=12)
            axes[i].axis('off')
        
        fig.suptitle('左手垂直抓取姿态演示 - 关键帧预览', fontsize=16)
        plt.tight_layout()
        plt.savefig('vertical_approach_preview.png', dpi=150, bbox_inches='tight')
        print("✅ 关键帧预览已保存: vertical_approach_preview.png")
        
        return gif_path, len(pil_frames)
    
    return None, 0

if __name__ == "__main__":
    try:
        gif_file, frame_count = create_vertical_approach_gif()
        if gif_file:
            print(f"\n🎬 成功创建GIF动画: {gif_file}")
            print(f"📊 总帧数: {frame_count}")
        else:
            print("❌ 未能创建GIF动画")
    except Exception as e:
        print(f"❌ 创建GIF时出错: {e}")
        import traceback
        traceback.print_exc()