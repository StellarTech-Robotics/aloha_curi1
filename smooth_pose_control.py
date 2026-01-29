#!/usr/bin/env python3
"""
平滑姿态控制 - 改进版
提供连贯的、可控速度的末端执行器姿态控制
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys, termios

# 导入基础控制功能
sys.path.append('./assets')
from curi1_control_posekeys import (
    model, data, find_arm_chain, TargetController, 
    quat_slerp, rpy_to_quat, quat_normalize, prompt_line
)

def minjerk(alpha):
    """最小加加速度插值 - S型平滑曲线"""
    return alpha**3 * (10 - 15*alpha + 6*alpha*alpha)

def smooth_move_to_pose(controller, side, target_pos, target_quat=None, 
                       duration=3.0, fps=50, use_minjerk=True, verbose=True):
    """
    改进的平滑姿态控制函数
    
    Args:
        controller: TargetController实例
        side: "left" 或 "right"
        target_pos: 目标位置 [x, y, z]
        target_quat: 目标四元数 [w, x, y, z] (可选)
        duration: 运动持续时间(秒) - 控制速度的关键参数
        fps: 插值帧率 (默认50fps，更平滑)
        use_minjerk: 是否使用S型加速度曲线
        verbose: 是否显示进度
    """
    if verbose:
        print(f"\n🎯 平滑控制 {side.upper()} 臂")
        print(f"   目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        if target_quat is not None:
            print(f"   目标姿态: [{target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f}]")
        else:
            print(f"   姿态控制: 仅位置")
        print(f"   运动时间: {duration:.1f}秒")
        print(f"   插值频率: {fps} FPS")
    
    # 参数计算
    steps = max(1, int(duration * fps))
    dt_step = duration / steps
    dt_sim = float(model.opt.timestep)
    
    # 获取起始状态
    c = controller.chains[side]
    start_pos = data.xpos[c["ee"]].copy()
    start_quat = data.xquat[c["ee"]].copy()
    
    # 如果没有指定目标姿态，保持当前姿态
    if target_quat is None:
        target_quat = start_quat.copy()
    
    if verbose:
        distance = np.linalg.norm(target_pos - start_pos)
        print(f"   移动距离: {distance*100:.1f}cm")
        print(f"   平均速度: {distance/duration*100:.1f}cm/s")
        print("   开始执行...")
    
    # 执行平滑轨迹
    start_time = time.perf_counter()
    
    for i in range(steps):
        step_start = time.perf_counter()
        
        # 计算插值参数
        alpha = (i + 1) / steps
        if use_minjerk:
            s = minjerk(alpha)  # S型曲线
        else:
            s = alpha  # 线性插值
        
        # 位置和姿态插值
        current_pos = (1.0 - s) * start_pos + s * target_pos
        current_quat = quat_slerp(start_quat, target_quat, s)
        
        # IK求解
        try:
            controller.set_target_from_ik_pose(side, current_pos, current_quat, 
                                             iters=15, damping=1e-4)
        except Exception as e:
            if verbose:
                print(f"   ⚠️ IK警告 step {i+1}: {e}")
            continue
        
        # 物理仿真步进
        sim_steps = max(1, int(round(dt_step / dt_sim)))
        for _ in range(sim_steps):
            controller.update_control()
            mujoco.mj_step(model, data)
        
        # 进度显示
        if verbose and i % max(1, steps//10) == 0:
            progress = (i + 1) / steps * 100
            elapsed = time.perf_counter() - start_time
            estimated_total = elapsed / alpha if alpha > 0 else duration
            remaining = max(0, estimated_total - elapsed)
            print(f"   进度: {progress:4.1f}% | 剩余: {remaining:.1f}s", end='\r')
        
        # 时间同步 (确保实时执行)
        step_elapsed = time.perf_counter() - step_start
        sleep_time = dt_step - step_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # 最终状态检查
    final_pos = data.xpos[c["ee"]].copy()
    final_error = np.linalg.norm(target_pos - final_pos)
    total_time = time.perf_counter() - start_time
    
    if verbose:
        print()  # 换行
        print(f"   ✅ 运动完成!")
        print(f"   实际耗时: {total_time:.2f}s (目标: {duration:.1f}s)")
        print(f"   最终误差: {final_error*1000:.1f}mm")
        print(f"   平均帧率: {steps/total_time:.1f} FPS")
    
    return final_error < 0.01  # 10mm误差内认为成功

def speed_controlled_demo():
    """演示不同速度的姿态控制"""
    print("🎮 速度控制演示")
    print("=" * 50)
    
    # 初始化
    chains = find_arm_chain()
    controller = TargetController(chains, kp=80.0, kd=15.0, ki=0.3)
    
    # 目标参数
    target_position = np.array([-0.12, -0.65, 0.65])  # 方块上方
    target_euler = [0, np.pi/12, 0]  # 轻微俯仰15度
    target_quat = rpy_to_quat(*target_euler)
    
    # 不同速度测试
    speed_tests = [
        {"duration": 1.0, "desc": "快速 (1秒)", "fps": 60},
        {"duration": 3.0, "desc": "正常 (3秒)", "fps": 50},
        {"duration": 5.0, "desc": "慢速 (5秒)", "fps": 40},
    ]
    
    print("\n测试不同运动速度:")
    for i, test in enumerate(speed_tests, 1):
        print(f"\n{i}. {test['desc']}")
        
        try:
            success = smooth_move_to_pose(
                controller, "right", target_position, target_quat,
                duration=test['duration'], fps=test['fps'], 
                use_minjerk=True, verbose=True
            )
            
            if success:
                print("   🎉 测试成功!")
            else:
                print("   ⚠️ 精度稍低，但运动完成")
                
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
        
        # 等待用户确认继续
        if i < len(speed_tests):
            input("   按Enter继续下一个测试...")
            print()

def interactive_speed_control():
    """交互式速度控制"""
    print("\n🎮 交互式速度控制")
    print("=" * 50)
    
    chains = find_arm_chain()
    controller = TargetController(chains, kp=100.0, kd=20.0, ki=0.5)
    
    print("输入目标位置和运动参数:")
    
    try:
        # 输入目标位置
        pos_input = input("目标位置 [x y z] (默认: -0.12 -0.65 0.65): ").strip()
        if pos_input:
            x, y, z = map(float, pos_input.split())
            target_pos = np.array([x, y, z])
        else:
            target_pos = np.array([-0.12, -0.65, 0.65])
        
        # 输入运动时间
        duration_input = input("运动时间(秒) (默认: 3.0): ").strip()
        duration = float(duration_input) if duration_input else 3.0
        
        # 输入机械臂选择
        side_input = input("机械臂 [left/right] (默认: right): ").strip()
        side = side_input.lower() if side_input in ['left', 'right'] else 'right'
        
        # 输入姿态参数
        pose_input = input("欧拉角 [roll pitch yaw] 度数 (回车跳过姿态控制): ").strip()
        if pose_input:
            roll, pitch, yaw = map(float, pose_input.split())
            target_quat = rpy_to_quat(np.radians(roll), np.radians(pitch), np.radians(yaw))
        else:
            target_quat = None
        
        # 执行控制
        print(f"\n执行控制...")
        success = smooth_move_to_pose(
            controller, side, target_pos, target_quat,
            duration=duration, fps=50, use_minjerk=True, verbose=True
        )
        
        if success:
            print("🎉 控制执行成功!")
        else:
            print("⚠️ 控制完成，但精度可能不够理想")
            
    except Exception as e:
        print(f"❌ 输入解析失败: {e}")

def enhanced_keyboard_control():
    """增强的键盘控制 (集成到现有系统)"""
    print("\n🎮 增强键盘控制 (集成版)")
    print("=" * 50)
    
    chains = find_arm_chain()
    controller = TargetController(chains, kp=100.0, kd=20.0, ki=0.5)
    
    mujoco.mj_forward(model, data)
    
    print("可用的速度预设:")
    speed_presets = {
        '1': {"duration": 1.0, "name": "极快", "fps": 60},
        '2': {"duration": 2.0, "name": "快速", "fps": 55}, 
        '3': {"duration": 3.0, "name": "正常", "fps": 50},
        '4': {"duration": 5.0, "name": "慢速", "fps": 45},
        '5': {"duration": 8.0, "name": "极慢", "fps": 40},
    }
    
    for key, preset in speed_presets.items():
        print(f"  {key} - {preset['name']} ({preset['duration']}秒)")
    
    print("\n输入格式: side x y z roll pitch yaw speed_preset")
    print("示例: right -0.12 -0.65 0.65 0 15 0 3")
    print("      (右臂, 位置, 俯仰15度, 正常速度)")
    
    try:
        line = prompt_line("\n输入控制参数 >>> ")
        parts = line.strip().split()
        
        if len(parts) < 4:
            print("❌ 参数不足")
            return
        
        # 解析参数
        side = parts[0].lower()
        x, y, z = map(float, parts[1:4])
        target_pos = np.array([x, y, z])
        
        # 姿态参数 (可选)
        target_quat = None
        if len(parts) >= 7:
            roll, pitch, yaw = map(float, parts[4:7])
            target_quat = rpy_to_quat(np.radians(roll), np.radians(pitch), np.radians(yaw))
            print(f"姿态: roll={roll}°, pitch={pitch}°, yaw={yaw}°")
        
        # 速度预设
        speed_key = parts[-1] if len(parts) >= 5 else '3'
        if speed_key in speed_presets:
            speed = speed_presets[speed_key]
            print(f"速度: {speed['name']} ({speed['duration']}秒)")
        else:
            speed = speed_presets['3']  # 默认正常速度
            print("使用默认正常速度")
        
        # 执行控制
        print(f"\n🚀 执行平滑控制...")
        success = smooth_move_to_pose(
            controller, side, target_pos, target_quat,
            duration=speed['duration'], fps=speed['fps'], 
            use_minjerk=True, verbose=True
        )
        
        if success:
            print("🎉 平滑控制成功!")
        
    except Exception as e:
        print(f"❌ 控制失败: {e}")

def main():
    """主程序"""
    print("🎯 平滑姿态控制系统")
    print("=" * 60)
    print("解决瞬间移动问题，提供连贯的速度可控运动")
    
    print("\n选择模式:")
    print("1. 速度控制演示")
    print("2. 交互式控制") 
    print("3. 增强键盘控制")
    print("4. 直接集成测试")
    
    try:
        choice = input("选择模式 [1-4]: ").strip()
        
        if choice == '1':
            speed_controlled_demo()
        elif choice == '2':
            interactive_speed_control()
        elif choice == '3':
            enhanced_keyboard_control()
        elif choice == '4':
            # 直接测试
            chains = find_arm_chain()
            controller = TargetController(chains)
            target_pos = np.array([-0.12, -0.65, 0.65])
            target_quat = rpy_to_quat(0, np.pi/12, 0)
            
            print("执行3秒平滑运动测试...")
            success = smooth_move_to_pose(
                controller, "right", target_pos, target_quat,
                duration=3.0, fps=50, use_minjerk=True
            )
            print(f"结果: {'成功' if success else '部分成功'}")
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n👋 用户退出")

if __name__ == "__main__":
    main()