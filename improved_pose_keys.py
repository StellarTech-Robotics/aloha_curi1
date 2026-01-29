#!/usr/bin/env python3
"""
改进版姿态按键控制 - 确保平滑连贯的运动
基于 curi1_control_posekeys.py，修复瞬间移动问题
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys, termios

# 导入基础设置
sys.path.append('./assets')

MODEL_PATH = "bimanual_curi1_transfer_cube.xml"

def minjerk(alpha):
    """最小加加速度插值 - S型平滑曲线"""
    return alpha**3 * (10 - 15*alpha + 6*alpha*alpha)

def prompt_line(prompt: str) -> str:
    """终端输入工具"""
    import sys, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] |= termios.ECHO | termios.ICANON
    termios.tcsetattr(fd, termios.TCSANOW, new)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
        return input(prompt)
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old)

# 四元数工具函数
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

def quat_slerp(q0, q1, alpha):
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        q = q0 + alpha*(q1 - q0)
        return quat_normalize(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0-alpha)*theta0)/sin0
    s1 = np.sin(alpha*theta0)/sin0
    return quat_normalize(s0*q0 + s1*q1)

def rpy_to_quat(roll, pitch, yaw):
    """欧拉角转四元数"""
    cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)
    cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
    cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return quat_normalize(np.array([w,x,y,z], dtype=np.float64))

# MuJoCo设置
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 导入控制器相关函数
from assets.curi1_control_posekeys import (
    find_arm_chain, TargetController, 
    set_gripper, print_gripper_state, debug_gripper_contacts,
    check_cube_proximity, set_joint_delta
)

def smooth_move_ee_to_pose(controller, side, target_pos, target_quat, 
                          seconds=3.0, fps=50, realtime=True, smooth=True, verbose=True):
    """
    改进的平滑末端执行器运动控制
    确保连贯运动，避免瞬间移动
    
    Args:
        controller: TargetController实例
        side: "left" 或 "right"
        target_pos: 目标位置 np.array([x,y,z])
        target_quat: 目标四元数 np.array([w,x,y,z])
        seconds: 运动持续时间 (控制速度的关键参数)
        fps: 插值帧率 (默认50，平滑度)
        realtime: 是否实时同步 (避免快速执行)
        smooth: 是否使用S型曲线
        verbose: 是否显示详细信息
    """
    if verbose:
        print(f"\n🎬 开始平滑运动控制")
        print(f"   机械臂: {side.upper()}")
        print(f"   目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"   目标姿态: [{target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f}]")
        print(f"   运动时间: {seconds:.1f}秒")
        print(f"   插值频率: {fps} FPS")
    
    steps = max(1, int(seconds * fps))
    dt_iter = seconds / steps
    dt_sim = float(model.opt.timestep)
    
    # 获取起始状态
    c = controller.chains[side]
    p0 = data.xpos[c["ee"]].copy()
    q0 = data.xquat[c["ee"]].copy()
    
    if verbose:
        distance = np.linalg.norm(target_pos - p0)
        print(f"   移动距离: {distance*100:.1f}cm")
        print(f"   平均速度: {distance/seconds*100:.1f}cm/s")
        print(f"   总插值点: {steps}")
        print("   开始执行运动...")
    
    start_time = time.perf_counter()
    
    # 执行插值运动
    for i in range(steps):
        t0 = time.perf_counter()
        
        alpha = (i + 1) / steps
        s = minjerk(alpha) if smooth else alpha  # S型或线性插值
        
        # 位置和姿态插值
        p = (1.0 - s) * p0 + s * target_pos
        q = quat_slerp(q0, target_quat, s)
        
        # IK求解并更新目标
        try:
            controller.set_target_from_ik_pose(side, p, q, iters=20, damping=1e-4)
        except Exception as e:
            if verbose and i % 10 == 0:
                print(f"   ⚠️ IK警告 step {i+1}: {e}")
        
        # 物理仿真步进
        n_sim = max(1, int(round(dt_iter / dt_sim)))
        for _ in range(n_sim):
            controller.update_control()
            mujoco.mj_step(model, data)
        
        # 进度显示
        if verbose and i % max(1, steps//10) == 0:
            progress = alpha * 100
            elapsed = time.perf_counter() - start_time
            remaining = (seconds - elapsed) if elapsed < seconds else 0
            print(f"   进度: {progress:4.1f}% | 剩余: {remaining:.1f}s", end='\r', flush=True)
        
        # 实时同步 - 关键！确保不会快速执行完成
        if realtime:
            elapsed = time.perf_counter() - t0
            sleep_time = dt_iter - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # 完成统计
    total_time = time.perf_counter() - start_time
    final_pos = data.xpos[c["ee"]].copy()
    final_error = np.linalg.norm(target_pos - final_pos)
    
    if verbose:
        print()  # 换行
        print(f"   ✅ 运动完成!")
        print(f"   实际耗时: {total_time:.2f}s (计划: {seconds:.1f}s)")
        print(f"   位置误差: {final_error*1000:.1f}mm")
        print(f"   实际帧率: {steps/total_time:.1f} FPS")
    
    return final_error < 0.015  # 15mm内认为成功

# 修改后的主控制循环
def main_with_smooth_controls():
    """带有平滑控制的主程序"""
    chains = find_arm_chain()
    controller = TargetController(chains, kp=100.0, kd=20.0, ki=0.5)
    
    print("=== Arm chain discovery ===")
    for s in ("left","right"):
        c = chains[s]
        jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in c["jids"]]
        print(f"{s.upper()}: base={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c['base'])} -> ee={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c['ee'])}")
        print("  joints:", jnames)
    
    mujoco.mj_forward(model, data)
    L_tgt = data.xpos[chains["left"]["ee"]].copy()
    R_tgt = data.xpos[chains["right"]["ee"]].copy()
    
    mode = "right"
    step = 0.01
    goto = True
    physics_paused = False
    
    # 速度预设
    speed_presets = {
        '1': 1.0,   # 极快
        '2': 2.0,   # 快速 
        '3': 3.0,   # 正常
        '4': 5.0,   # 慢速
        '5': 8.0,   # 极慢
    }
    
    HELP = f"""
[改进版姿态控制] (keep TERMINAL focused)
  h    : help
  q    : quit
  0    : reset to keyframe 0
  1/2/3: LEFT / RIGHT / BOTH
  
  === 平滑姿态控制 (NEW!) ===
  O    : 位置 + 欧拉角 (平滑运动)
  Q    : 位置 + 四元数 (平滑运动)
  I    : 仅位置控制 (平滑运动)
  
  格式示例:
  O键: right -0.12 -0.65 0.65 0 15 0 3
       (右臂, 位置, roll=0°, pitch=15°, yaw=0°, 时间=3秒)
  
  速度预设: {speed_presets}
  
  === 基础控制 ===
  w/s, a/d, r/f : LEFT arm +Y/-Y, -X/+X, +Z/-Z
  i/k, j/l, u/h : RIGHT arm +Y/-Y, -X/+X, +Z/-Z
  z/x, n/m      : gripper control
  P             : pause physics
"""
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(2)
        import os, select, tty
        from contextlib import contextmanager
        
        @contextmanager
        def raw_terminal_mode(file):
            fd = file.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd); yield
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        
        def read_keys(timeout=0.0):
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if not r: return []
            data_bytes = os.read(sys.stdin.fileno(), 1024)
            return [chr(c) for c in data_bytes]
        
        print(HELP)
        print("\n🎬 现在支持平滑运动控制！")
        print("试试按 'O' 键输入: right -0.12 -0.65 0.65 0 15 0 3")
        print("(右臂移动到方块上方，俯仰15度，3秒完成)")
        
        with raw_terminal_mode(sys.stdin):
            while viewer.is_running():
                for ch in read_keys(0.0):
                    if ch == 'h':
                        print(HELP)
                    elif ch == 'q':
                        return
                    elif ch == '0':
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                        data.qvel[:] = 0; data.act[:] = 0
                        mujoco.mj_forward(model, data)
                        L_tgt = data.xpos[chains["left"]["ee"]].copy()
                        R_tgt = data.xpos[chains["right"]["ee"]].copy()
                        print("[reset] keyframe 0")
                    elif ch == '1':
                        mode = "left"; print("[mode] LEFT")
                    elif ch == '2':
                        mode = "right"; print("[mode] RIGHT")
                    elif ch == '3':
                        mode = "both"; print("[mode] BOTH")
                    elif ch == 'P':
                        physics_paused = not physics_paused
                        print(f"[physics] {'PAUSED' if physics_paused else 'RUNNING'}")
                    
                    # ===== 平滑姿态控制触发 =====
                    elif ch == 'O':  # 欧拉角控制
                        try:
                            print("\n🎬 平滑欧拉角姿态控制")
                            line = prompt_line(
                                "格式: side x y z roll pitch yaw(deg) duration(s)\n"
                                "例如: right -0.12 -0.65 0.65 0 15 0 3\n> "
                            )
                            parts = line.strip().split()
                            if len(parts) < 4:
                                print("❌ 参数不足")
                                continue
                                
                            side, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                            pos = np.array([x, y, z], dtype=np.float64)
                            
                            # 姿态参数 (可选)
                            if len(parts) >= 7:
                                r, p_, yw = float(parts[4]), float(parts[5]), float(parts[6])
                                quat = rpy_to_quat(np.deg2rad(r), np.deg2rad(p_), np.deg2rad(yw))
                                print(f"目标姿态: roll={r}°, pitch={p_}°, yaw={yw}°")
                            else:
                                quat = data.xquat[chains[side.lower()]["ee"]].copy()  # 保持当前姿态
                                print("保持当前姿态")
                            
                            # 运动时间
                            duration = float(parts[7]) if len(parts) >= 8 else 3.0
                            
                            print(f"🎬 执行平滑运动: {side} 臂, {duration}秒")
                            success = smooth_move_ee_to_pose(
                                controller, side.lower(), pos, quat, 
                                seconds=duration, fps=50, realtime=True, smooth=True, verbose=True
                            )
                            
                            if success:
                                print("🎉 平滑运动完成!")
                            else:
                                print("⚠️ 运动完成，但可能有精度偏差")
                                
                        except Exception as e:
                            print(f"❌ 控制失败: {e}")
                    
                    elif ch == 'Q':  # 四元数控制
                        try:
                            print("\n🎬 平滑四元数姿态控制")
                            line = prompt_line(
                                "格式: side x y z qw qx qy qz duration(s)\n"
                                "例如: left -0.2 -0.6 0.6 0.966 0 0.259 0 2\n> "
                            )
                            parts = line.strip().split()
                            if len(parts) < 4:
                                print("❌ 参数不足")
                                continue
                                
                            side, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                            pos = np.array([x, y, z], dtype=np.float64)
                            
                            if len(parts) >= 8:
                                qw, qx, qy, qz = map(float, parts[4:8])
                                quat = quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))
                            else:
                                quat = data.xquat[chains[side.lower()]["ee"]].copy()
                            
                            duration = float(parts[8]) if len(parts) >= 9 else 3.0
                            
                            print(f"🎬 执行平滑运动: {side} 臂, {duration}秒")
                            success = smooth_move_ee_to_pose(
                                controller, side.lower(), pos, quat,
                                seconds=duration, fps=50, realtime=True, smooth=True, verbose=True
                            )
                            
                        except Exception as e:
                            print(f"❌ 控制失败: {e}")
                    
                    elif ch == 'I':  # 仅位置控制
                        try:
                            print("\n🎬 平滑位置控制")
                            line = prompt_line(
                                "格式: side x y z duration(s)\n"
                                "例如: right -0.12 -0.65 0.65 2\n> "
                            )
                            parts = line.strip().split()
                            if len(parts) < 4:
                                print("❌ 参数不足")
                                continue
                                
                            side, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                            pos = np.array([x, y, z])
                            duration = float(parts[4]) if len(parts) >= 5 else 3.0
                            
                            # 保持当前姿态
                            current_quat = data.xquat[chains[side.lower()]["ee"]].copy()
                            
                            print(f"🎬 执行平滑位置移动: {side} 臂, {duration}秒")
                            success = smooth_move_ee_to_pose(
                                controller, side.lower(), pos, current_quat,
                                seconds=duration, fps=50, realtime=True, smooth=True, verbose=True
                            )
                            
                        except Exception as e:
                            print(f"❌ 控制失败: {e}")
                    
                    # ===== 基础控制 (保持原有功能) =====
                    elif ch == 'w': L_tgt[1] += step if mode in ('left','both') else 0
                    elif ch == 's': L_tgt[1] -= step if mode in ('left','both') else 0
                    elif ch == 'a': L_tgt[0] -= step if mode in ('left','both') else 0
                    elif ch == 'd': L_tgt[0] += step if mode in ('left','both') else 0
                    elif ch == 'r': L_tgt[2] += step if mode in ('left','both') else 0
                    elif ch == 'f': L_tgt[2] -= step if mode in ('left','both') else 0
                    
                    elif ch == 'i': R_tgt[1] += step if mode in ('right','both') else 0
                    elif ch == 'k': R_tgt[1] -= step if mode in ('right','both') else 0
                    elif ch == 'j': R_tgt[0] -= step if mode in ('right','both') else 0
                    elif ch == 'l': R_tgt[0] += step if mode in ('right','both') else 0
                    elif ch == 'u': R_tgt[2] += step if mode in ('right','both') else 0
                    elif ch == 'h': R_tgt[2] -= step if mode in ('right','both') else 0
                    
                    # 夹爪控制
                    elif ch == 'z': 
                        set_gripper("left", -0.005, controller)
                        print_gripper_state()
                    elif ch == 'x': 
                        set_gripper("left", +0.005, controller) 
                        print_gripper_state()
                    elif ch == 'n': 
                        set_gripper("right", -0.005, controller)
                        print_gripper_state()
                    elif ch == 'm': 
                        set_gripper("right", +0.005, controller)
                        print_gripper_state()
                
                # 基础IK控制更新
                if goto and not physics_paused:
                    # 这里使用简化的IK而不是平滑运动，用于实时控制
                    from assets.curi1_control_posekeys import ik_step_dynamic
                    if mode in ('left','both'):
                        ik_step_dynamic(controller, 'left', L_tgt, max_iters=1)
                    if mode in ('right','both'):
                        ik_step_dynamic(controller, 'right', R_tgt, max_iters=1)
                
                # 物理步进
                if not physics_paused:
                    controller.update_control()
                    mujoco.mj_step(model, data)
                
                viewer.sync()
                time.sleep(0.001)

if __name__ == "__main__":
    print("🎬 改进版平滑姿态控制系统")
    print("=" * 60)
    print("解决瞬间移动问题，提供真正的连贯运动")
    print("支持速度控制的平滑位置和姿态运动")
    
    try:
        main_with_smooth_controls()
    except KeyboardInterrupt:
        print("\n👋 用户退出")
        pass