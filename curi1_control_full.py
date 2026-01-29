#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# WSL2环境下解决OpenCV的Qt线程问题
import os
import sys
import warnings
import json
from pathlib import Path

# 强制使用xcb（X11）后端，避免offscreen插件的线程问题
# 必须在导入cv2之前设置
if 'QT_QPA_PLATFORM' in os.environ:
    # 如果已经设置了offscreen或其他有问题的值，删除它
    del os.environ['QT_QPA_PLATFORM']

# 如果有DISPLAY，强制使用xcb
if 'DISPLAY' in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    # 禁用Qt的所有警告和调试输出
    os.environ['QT_LOGGING_RULES'] = '*=false'
    os.environ['QT_DEBUG_PLUGINS'] = '0'

# 抑制Qt线程警告（重定向stderr）
class SuppressQtWarnings:
    """上下文管理器：临时抑制Qt的stderr警告"""
    def __init__(self):
        self.null_fd = None
        self.old_stderr = None

    def __enter__(self):
        # 只抑制Qt线程警告，保留其他错误
        return self

    def __exit__(self, *args):
        pass

# 全局抑制Qt moveToThread警告
def _suppress_qt_warnings():
    """过滤stderr中的Qt moveToThread警告"""
    import io
    class FilteredStderr(io.TextIOBase):
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
            self.buffer = ""

        def write(self, text):
            # 过滤掉Qt线程警告
            if "QObject::moveToThread" not in text and "Cannot move to target thread" not in text:
                return self.original_stderr.write(text)
            return len(text)

        def flush(self):
            return self.original_stderr.flush()

    sys.stderr = FilteredStderr(sys.stderr)

import mujoco
import mujoco.viewer
import numpy as np
import time

# === EE pose utils (minimal, read-only) ===
def quat_to_rpy(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) -> roll,pitch,yaw (radians), convention Rz*Ry*Rx."""
    # roll (x)
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y)
    sinp = 2.0 * (qw*qy - qz*qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def get_ee_pose(controller, side):
    """Read current EE pose in world frame: (pos[m], quat[w,x,y,z])."""
    ee = controller.chains[side]["ee"]
    pos = data.xpos[ee].copy()
    quat = data.xquat[ee].copy()
    return pos, quat

def print_ee_pose(controller, side=None):
    """Print pos + quat + rpy(deg); side in {'left','right'} or None for both."""
    sides = ["left", "right"] if side is None else [side]
    for s in sides:
        try:
            pos, quat = get_ee_pose(controller, s)
            r, p, y = quat_to_rpy(quat[0], quat[1], quat[2], quat[3])
            r_d, p_d, y_d = np.rad2deg([r, p, y])
            print(
                f"[{s}] pos(m): [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
                f"quat(wxyz): [{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]  "
                f"rpy(deg): [{r_d:+.1f}, {p_d:+.1f}, {y_d:+.1f}]"
            )
        except Exception as e:
            print(f"[{s}] 读取EE位姿失败: {e}")

# —— 让终端临时恢复“可见输入”的小工具 ——
def prompt_line(prompt: str) -> str:
    import sys, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    # 打开回显(ECHO) + 行编辑(ICANON)
    new = termios.tcgetattr(fd)
    new[3] |= termios.ECHO | termios.ICANON
    termios.tcsetattr(fd, termios.TCSANOW, new)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)  # 清掉之前按键残留
        return input(prompt)
    finally:
        # 恢复为原先的raw/noecho（供你的 getch/按键循环继续使用）
        termios.tcsetattr(fd, termios.TCSANOW, old)


# === Quaternion & SO(3) helpers ===
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_error_vec(q_des, q_cur):
    """Small-angle orientation error vector (world frame)."""
    qd = quat_normalize(q_des)
    qc = quat_normalize(q_cur)
    dq = quat_mul(qd, quat_conj(qc))
    if dq[0] < 0:  # choose shortest path
        dq = -dq
    return 2.0 * dq[1:4]

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
    """XYZ (roll, pitch, yaw in rad) -> quaternion (w,x,y,z)."""
    cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)
    cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
    cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return quat_normalize(np.array([w,x,y,z], dtype=np.float64))

MODEL_PATH = "assets/bimanual_curi1_transfer_cube.xml"

# ===== MuJoCo setup =====
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("CONTACT disabled? ", bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT))

# ===== Robot names =====
LEFT_BASE_BODY  = "l_base_link1"
RIGHT_BASE_BODY = "r_base_link1"
LEFT_EE_BODY    = "l_rmg42_base_link"
RIGHT_EE_BODY   = "r_rmg42_base_link"
LEFT_JOINT_PREFIX  = "l_joint"
RIGHT_JOINT_PREFIX = "r_joint"
LEFT_GRIPPER_JOINTS  = ["Joint_finger1", "Joint_finger2"]
RIGHT_GRIPPER_JOINTS = ["r_Joint_finger1", "r_Joint_finger2"]

# Trajectory timing scale (>1 slows down motions, <1 speeds up)
TRAJECTORY_TIME_SCALE = 1.0
# Fraction of trajectory time used for acceleration/deceleration (0~0.5)
TRAJECTORY_ACCEL_RATIO = 0.02
TRAJECTORY_DECEL_RATIO = 0.02

def mj_id(objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return -1

def find_arm_chain():
    left, right = [], []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.startswith(LEFT_JOINT_PREFIX):
            left.append((j, name))
        elif name.startswith(RIGHT_JOINT_PREFIX):
            right.append((j, name))
    def sort6(pairs):
        def k(p):
            j, name = p
            num = "".join([c for c in name if c.isdigit()])
            return int(num) if num else 999
        return [j for j,_ in sorted(pairs, key=k)][:6]
    l_ids = sort6(left)
    r_ids = sort6(right)
    l_qadr = [model.jnt_qposadr[j] for j in l_ids]
    r_qadr = [model.jnt_qposadr[j] for j in r_ids]
    return {
        "left":  {"base": mj_id(mujoco.mjtObj.mjOBJ_BODY, LEFT_BASE_BODY),  "ee": mj_id(mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY),  "jids": l_ids, "qadr": l_qadr},
        "right": {"base": mj_id(mujoco.mjtObj.mjOBJ_BODY, RIGHT_BASE_BODY), "ee": mj_id(mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY), "jids": r_ids, "qadr": r_qadr},
    }

def print_chain(ch):
    print("=== Arm chain discovery ===")
    for s in ("left","right"):
        c = ch[s]
        jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in c["jids"]]
        print(f"{s.upper()}: base={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c['base'])} -> ee={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c['ee'])}")
        print("  joints:", jnames)

def extract_14dim_qpos(data_qpos, chains):
    """
    从CURI的完整qpos中提取14维Mobile ALOHA兼容格式
    格式: [左臂6, 左夹爪1, 右臂6, 右夹爪1]
    """
    result = np.zeros(14, dtype=np.float32)
    
    # 左臂6维 (索引0-5)
    left_chain = chains["left"]
    for i, qadr in enumerate(left_chain["qadr"]):
        result[i] = data_qpos[qadr]
    
    # 左夹爪1维 (索引6) - 取两个手指的平均值
    try:
        left_finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger1")
        left_finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger2")
        left_finger1_qadr = model.jnt_qposadr[left_finger1_id]
        left_finger2_qadr = model.jnt_qposadr[left_finger2_id]
        result[6] = (data_qpos[left_finger1_qadr] + data_qpos[left_finger2_qadr]) / 2.0
    except:
        result[6] = 0.0  # 如果找不到夹爪关节，设为0
    
    # 右臂6维 (索引7-12)
    right_chain = chains["right"]
    for i, qadr in enumerate(right_chain["qadr"]):
        result[7 + i] = data_qpos[qadr]
    
    # 右夹爪1维 (索引13) - 取两个手指的平均值
    try:
        right_finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger1")
        right_finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger2")
        right_finger1_qadr = model.jnt_qposadr[right_finger1_id]
        right_finger2_qadr = model.jnt_qposadr[right_finger2_id]
        result[13] = (data_qpos[right_finger1_qadr] + data_qpos[right_finger2_qadr]) / 2.0
    except:
        result[13] = 0.0  # 如果找不到夹爪关节，设为0
    
    return result

def expand_14dim_to_full_qpos(mobile_aloha_qpos, current_full_qpos, chains):
    """
    将14维Mobile ALOHA格式扩展为CURI的完整qpos格式
    用于从Mobile ALOHA模型的输出恢复完整的机器人状态
    """
    result = current_full_qpos.copy()
    
    # 左臂6维
    left_chain = chains["left"]
    for i, qadr in enumerate(left_chain["qadr"]):
        result[qadr] = mobile_aloha_qpos[i]
    
    # 左夹爪 - 将1维分配给两个手指
    try:
        left_finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger1")
        left_finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger2")
        left_finger1_qadr = model.jnt_qposadr[left_finger1_id]
        left_finger2_qadr = model.jnt_qposadr[left_finger2_id]
        gripper_val = mobile_aloha_qpos[6]
        result[left_finger1_qadr] = gripper_val
        result[left_finger2_qadr] = -gripper_val  # 相反方向
    except:
        pass
    
    # 右臂6维
    right_chain = chains["right"]
    for i, qadr in enumerate(right_chain["qadr"]):
        result[qadr] = mobile_aloha_qpos[7 + i]
    
    # 右夹爪 - 将1维分配给两个手指
    try:
        right_finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger1")
        right_finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger2")
        right_finger1_qadr = model.jnt_qposadr[right_finger1_id]
        right_finger2_qadr = model.jnt_qposadr[right_finger2_id]
        gripper_val = mobile_aloha_qpos[13]
        result[right_finger1_qadr] = gripper_val
        result[right_finger2_qadr] = -gripper_val  # 相反方向
    except:
        pass
    
    return result

# ===== 找到关节对应的执行器 =====
def find_joint_actuators():
    """建立关节到执行器的映射"""
    joint_to_actuator = {}
    for aid in range(model.nu):
        # 获取执行器作用的关节
        if model.actuator_trntype[aid] == 0:  # joint transmission
            jid = model.actuator_trnid[aid, 0]
            if jid >= 0:
                joint_to_actuator[jid] = aid
    return joint_to_actuator

joint_to_actuator = find_joint_actuators()

# ===== 夹爪函数 =====
def _find_actuator_ids():
    names = {
        "left":  ["Joint_finger1", "Joint_finger2"],
        "right": ["r_Joint_finger1", "r_Joint_finger2"],
    }
    out = {"left": [], "right": []}
    for side, lst in names.items():
        ids = []
        for nm in lst:
            try:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except Exception:
                aid = -1
            ids.append(aid)
        out[side] = ids
    return out

_GRIP_ACT = _find_actuator_ids()

# ===== 碰撞检测和调试函数 =====
def debug_gripper_contacts(side="left"):
    """检查夹爪是否与物体发生碰撞"""
    finger_names = ["Joint_finger1", "Joint_finger2"] if side == "left" else ["r_Joint_finger1", "r_Joint_finger2"]
    
    print(f"\n=== {side.upper()} 夹爪碰撞检测 ===")
    
    # 检查每个手指的碰撞
    for finger_name in finger_names:
        try:
            finger_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
            if finger_jid >= 0:
                finger_body_id = model.jnt_bodyid[finger_jid]
                finger_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, finger_body_id)
                print(f"手指 {finger_name} -> body: {finger_body_name}")
        except:
            continue
    
    # 检查当前所有碰撞
    contact_count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        
        # 检查是否涉及夹爪
        is_gripper_contact = any(finger in geom1_name or finger in geom2_name 
                                for finger in ["finger", "gripper", side[0] + "_"])
        
        if is_gripper_contact:
            contact_count += 1
            force_norm = np.linalg.norm(contact.frame[:3])  # 接触力的大小
            print(f"碰撞 {contact_count}: {geom1_name} <-> {geom2_name}, 力度: {force_norm:.3f}")
    
    if contact_count == 0:
        print("无夹爪相关碰撞")
    
    return contact_count

def check_cube_proximity(side="left"):
    """检查夹爪是否接近方块"""
    try:
        # 尝试找到方块物体
        cube_id = -1
        for name in ["object", "cube", "box", "block", "target"]:
            cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if cube_id >= 0:
                break
        
        if cube_id < 0:
            print("未找到方块物体")
            return False, 999.0
        
        cube_pos = data.xpos[cube_id].copy()
        
        # 获取夹爪位置
        ee_name = "l_rmg42_base_link" if side == "left" else "r_rmg42_base_link"
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if ee_id < 0:
            return False, 999.0
        
        gripper_pos = data.xpos[ee_id].copy()
        distance = np.linalg.norm(cube_pos - gripper_pos)
        
        is_close = distance < 0.1  # 10cm内认为接近
        print(f"[{side}] 夹爪距离方块: {distance:.3f}m {'(接近)' if is_close else ''}")
        
        return is_close, distance
        
    except Exception as e:
        print(f"检查方块距离时出错: {e}")
        return False, 999.0

def print_enhanced_gripper_state():
    """显示详细的夹爪状态信息"""
    def analyze_side(side):
        jnames = ["Joint_finger1","Joint_finger2"] if side=="left" else ["r_Joint_finger1","r_Joint_finger2"]
        positions = []
        forces = []
        
        for jn in jnames:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                adr = model.jnt_qposadr[jid]
                dadr = model.jnt_dofadr[jid]
                
                pos = float(data.qpos[adr])
                vel = float(data.qvel[dadr]) if dadr >= 0 else 0.0
                
                positions.append(pos)
                
                # 尝试获取关节力
                if hasattr(data, 'qfrc_constraint') and dadr >= 0:
                    force = float(data.qfrc_constraint[dadr])
                    forces.append(force)
                else:
                    forces.append(0.0)
                    
            except Exception:
                positions.append(None)
                forces.append(0.0)
        
        # 获取执行器控制值
        ctrl_values = []
        for aid in _GRIP_ACT.get(side, []):
            if aid is not None and aid >= 0:
                ctrl_values.append(float(data.ctrl[aid]))
            else:
                ctrl_values.append(None)
        
        return positions, forces, ctrl_values
    
    l_pos, l_force, l_ctrl = analyze_side("left")
    r_pos, r_force, r_ctrl = analyze_side("right")
    
    print(f"[gripper详细] LEFT:  pos={np.round(l_pos,4)} force={np.round(l_force,2)} ctrl={np.round(l_ctrl,4)}")
    print(f"[gripper详细] RIGHT: pos={np.round(r_pos,4)} force={np.round(r_force,2)} ctrl={np.round(r_ctrl,4)}")

# ===== 修改后的目标跟踪控制器（支持夹爪手动控制）=====
class TargetController:
    """用于平滑跟踪目标位置的控制器，支持夹爪手动控制"""
    def __init__(self, chains, kp=50.0, kd=10.0, ki=0.5):
        self.chains = chains
        self.kp = kp  # 比例增益
        self.kd = kd  # 微分增益  
        self.ki = ki  # 积分增益
        self.joint_to_actuator = joint_to_actuator
        
        # 目标关节位置
        self.target_qpos = {}
        for side in ["left", "right"]:
            self.target_qpos[side] = np.zeros(len(chains[side]["qadr"]))
            # 初始化为当前位置
            for i, qadr in enumerate(chains[side]["qadr"]):
                self.target_qpos[side][i] = data.qpos[qadr]
        
        # 积分误差
        self.integral_error = {"left": np.zeros(6), "right": np.zeros(6)}
        
        # 夹爪目标
        self.gripper_targets = {"left": 0.0, "right": 0.0}

        # ===== 添加夹爪手动控制支持 =====
        self.gripper_manual_mode = {"left": False, "right": False}
        self.gripper_hold_positions = {"left": None, "right": None}

        # ===== 添加位置伺服锁定模式 =====
        self.gripper_servo_lock = {"left": False, "right": False}  # 位置伺服锁定状态
        self.gripper_lock_position = {"left": None, "right": None}  # 锁定的目标位置
        self.gripper_servo_kp = 200.0  # 位置伺服增益（模仿ACT++）

        # ===== 恒定夹紧力模式 =====
        self.gripper_force_lock = {"left": False, "right": False}
        self.gripper_force_target = {"left": 0.0, "right": 0.0}
        self.gripper_force_dofs = {"left": [], "right": []}
        for side, names in {
            "left": ["Joint_finger1", "Joint_finger2"],
            "right": ["r_Joint_finger1", "r_Joint_finger2"],
        }.items():
            dofs = []
            for nm in names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                if jid >= 0:
                    dofs.append(model.jnt_dofadr[jid])
                else:
                    dofs.append(-1)
            self.gripper_force_dofs[side] = dofs
        
    def set_target_from_ik(self, side, target_pos, pos_tol=1e-4, max_iters=80):
        """使用IK计算目标关节角度"""
        c = self.chains[side]
        
        # 多次迭代IK以获得更准确的解
        temp_qpos = data.qpos.copy()
        for _ in range(max_iters):
            cur = data.xpos[c["ee"]].copy()
            e = target_pos - cur
            err = np.linalg.norm(e)
            if err < pos_tol:
                break
                
            J = self.numeric_jac(side)
            JJt = J @ J.T
            reg = max(5e-6, min(3e-4, err * 0.4))
            dq = J.T @ np.linalg.solve(JJt + (reg**2)*np.eye(3), e)
            clip = 0.04 if err > 1e-2 else (0.03 if err > 3e-3 else 0.015)
            dq = np.clip(dq, -clip, clip)
            
            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                v = data.qpos[qadr] + dq[i]
                rmin, rmax = model.jnt_range[jid]
                if rmin < rmax:
                    v = max(rmin, min(rmax, v))
                data.qpos[qadr] = v
            mujoco.mj_forward(model, data)
        
        # 保存目标位置
        for i, qadr in enumerate(c["qadr"]):
            self.target_qpos[side][i] = data.qpos[qadr]
        
        # 恢复原始状态
        data.qpos[:] = temp_qpos
        mujoco.mj_forward(model, data)
    
    def numeric_jac_pose(self, side, eps=1e-5):
        """Compute 6xN numeric Jacobian (pos + ori small-angle) for the EE."""
        c = self.chains[side]
        n = len(c["qadr"])
        Jp = np.zeros((3, n), dtype=np.float64)
        Jo = np.zeros((3, n), dtype=np.float64)
        qbackup = data.qpos.copy()
        mujoco.mj_forward(model, data)

        p0 = data.xpos[c["ee"]].copy()
        q0 = data.xquat[c["ee"]].copy()

        for k, adr in enumerate(c["qadr"]):
            data.qpos[:] = qbackup
            data.qpos[adr] += eps
            mujoco.mj_forward(model, data)
            p_plus = data.xpos[c["ee"]].copy()
            q_plus = data.xquat[c["ee"]].copy()

            data.qpos[:] = qbackup
            data.qpos[adr] -= eps
            mujoco.mj_forward(model, data)
            p_minus = data.xpos[c["ee"]].copy()
            q_minus = data.xquat[c["ee"]].copy()

            Jp[:, k] = (p_plus - p_minus) / (2.0*eps)
            e_plus  = quat_error_vec(q_plus, q0)
            e_minus = quat_error_vec(q_minus, q0)
            Jo[:, k] = (e_plus - e_minus) / (2.0*eps)

        data.qpos[:] = qbackup
        mujoco.mj_forward(model, data)
        return np.vstack([Jp, Jo])

    def set_target_from_ik_pose(self, side, target_pos, target_quat, iters=90, damping=5e-5, pos_tol=1.5e-4, ori_tol=3e-3):
        """IK solve for both position and orientation. target_quat is (w,x,y,z)."""
        c = self.chains[side]
        temp_qpos = data.qpos.copy()

        for _ in range(iters):
            cur_p = data.xpos[c["ee"]].copy()
            cur_q = data.xquat[c["ee"]].copy()
            e_pos = target_pos - cur_p
            e_ori = quat_error_vec(target_quat, cur_q)
            err_pos = np.linalg.norm(e_pos)
            err_ori = np.linalg.norm(e_ori)
            if err_pos < pos_tol and err_ori < ori_tol:
                break

            J = self.numeric_jac_pose(side)
            JT = J.T
            reg = max(5e-6, min(3e-4, err_pos * 0.35))
            H = JT @ J + (max(reg, damping)**2) * np.eye(J.shape[1])
            g = JT @ np.hstack([e_pos, e_ori])
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, np.hstack([e_pos, e_ori]), rcond=None)

            clip = 0.035 if err_pos > 3e-3 else 0.015
            dq = np.clip(dq, -clip, clip)
            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                v = float(data.qpos[qadr] + dq[i])
                # respect hinge/slide limits if present
                rmin, rmax = model.jnt_range[jid]
                if rmin < rmax:
                    v = max(rmin, min(rmax, v))
                data.qpos[qadr] = v
            mujoco.mj_forward(model, data)

        for i, qadr in enumerate(c["qadr"]):
            self.target_qpos[side][i] = data.qpos[qadr]

        data.qpos[:] = temp_qpos
        mujoco.mj_forward(model, data)

    def numeric_jac(self, side, eps=1e-5):
        """计算数值雅可比矩阵"""
        c = self.chains[side]
        J = np.zeros((3, len(c["qadr"])))
        qbackup = data.qpos.copy()
        mujoco.mj_forward(model, data)
        
        for k, adr in enumerate(c["qadr"]):
            data.qpos[:] = qbackup
            data.qpos[adr] += eps
            mujoco.mj_forward(model, data)
            p_plus = data.xpos[c["ee"]].copy()
            
            data.qpos[:] = qbackup
            data.qpos[adr] -= eps
            mujoco.mj_forward(model, data)
            p_minus = data.xpos[c["ee"]].copy()
            
            J[:,k] = (p_plus - p_minus) / (2*eps)
        
        data.qpos[:] = qbackup
        mujoco.mj_forward(model, data)
        return J
    
    # ===== 添加夹爪手动控制方法 =====
    def set_gripper_manual_mode(self, side, enabled, hold_position=None):
        """设置夹爪手动控制模式"""
        self.gripper_manual_mode[side] = enabled
        if enabled and hold_position is not None:
            self.gripper_hold_positions[side] = hold_position
        elif not enabled:
            self.gripper_hold_positions[side] = None
            self.gripper_force_lock[side] = False
            self.gripper_force_target[side] = 0.0
    
    def update_gripper_hold_position(self, side):
        """更新夹爪保持位置为当前位置"""
        if not self.gripper_manual_mode[side]:
            return

        jnames = ["Joint_finger1","Joint_finger2"] if side=="left" else ["r_Joint_finger1","r_Joint_finger2"]
        for jn in jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                adr = model.jnt_qposadr[jid]
                self.gripper_hold_positions[side] = float(data.qpos[adr])
                break

    def toggle_gripper_servo_lock(self, side):
        """
        切换夹爪位置伺服锁定模式

        工作原理（模仿ACT++）：
        1. 锁定当前夹爪位置作为目标
        2. 每步计算：force = kp × (target_pos - current_pos)
        3. 如果手指被物体阻挡，位置误差持续存在
        4. 因此持续输出力，实现抓取效果

        这与MuJoCo的position actuator原理相同
        """
        self.gripper_servo_lock[side] = not self.gripper_servo_lock[side]

        if self.gripper_servo_lock[side]:
            # 读取当前夹爪位置并锁定
            jnames = ["Joint_finger1", "Joint_finger2"] if side == "left" else ["r_Joint_finger1", "r_Joint_finger2"]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnames[0])
            if jid >= 0:
                adr = model.jnt_qposadr[jid]
                current_pos = data.qpos[adr]
                self.gripper_lock_position[side] = current_pos
                print(f"[SERVO LOCK {side.upper()}] 已锁定位置: {current_pos:.4f}m")
                print(f"[SERVO LOCK {side.upper()}] 使用 kp={self.gripper_servo_kp} 的位置伺服控制")
                print(f"[提示] {side}夹爪已锁定当前位置，将持续施加抓取力")
                print(f"[提示] 再次按 Ctrl+{'G' if side=='left' else 'H'} 解除锁定")
        else:
            # 解除锁定
            self.gripper_lock_position[side] = None
            print(f"[SERVO LOCK {side.upper()}] 已解锁，恢复手动控制")
            print(f"[提示] {side}夹爪已解除锁定")

        return self.gripper_servo_lock[side]

    def toggle_gripper_force_lock(self, side, force_newton=10.0):
        """
        切换夹爪恒定夹紧力模式。
        force_newton: 施加在每根手指滑块上的力（正值表示向内夹紧）。
        """
        self.gripper_force_lock[side] = not self.gripper_force_lock[side]

        if self.gripper_force_lock[side]:
            # 解除位置伺服锁定，避免相互冲突
            self.gripper_servo_lock[side] = False
            self.gripper_lock_position[side] = None

            # 记录当前夹爪位置，方便继续使用手动模式时不抖动
            self.set_gripper_manual_mode(side, True)
            self.update_gripper_hold_position(side)

            self.gripper_force_target[side] = abs(float(force_newton))
            print(f"[FORCE LOCK {side.upper()}] 已启用恒定夹紧力 {self.gripper_force_target[side]:.2f} N")
            print(f"[FORCE LOCK {side.upper()}] 将持续向内施压，Ctrl+{'G' if side=='left' else 'H'} 再次按下可解除")
        else:
            self.gripper_force_target[side] = 0.0
            print(f"[FORCE LOCK {side.upper()}] 已解除恒定夹紧力，恢复手动控制")

        return self.gripper_force_lock[side]
    
    def update_control(self):
        """更新控制信号（PID控制）"""
        for side in ["left", "right"]:
            c = self.chains[side]
            
            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                # 计算误差
                error = self.target_qpos[side][i] - data.qpos[qadr]
                
                # 计算速度（用于D项）
                dadr = model.jnt_dofadr[jid]
                vel = data.qvel[dadr] if dadr >= 0 else 0.0
                
                # 更新积分误差
                self.integral_error[side][i] += error * 0.001  # dt约为0.001
                self.integral_error[side][i] = np.clip(self.integral_error[side][i], -0.5, 0.5)
                
                # PID控制
                control = (self.kp * error - self.kd * vel + 
                          self.ki * self.integral_error[side][i])
                
                # 设置执行器控制
                if jid in self.joint_to_actuator:
                    aid = self.joint_to_actuator[jid]
                    # 限制控制信号
                    clo, chi = model.actuator_ctrlrange[aid]
                    # 对于位置控制器，直接设置目标位置
                    data.ctrl[aid] = self.target_qpos[side][i]
                    # 对于力矩控制器，使用PID输出
                    # data.ctrl[aid] = np.clip(control, clo, chi)
        
        # 更新夹爪控制（修改后的版本）
        self.update_gripper_control()
        self.apply_gripper_force_lock()

    def update_gripper_control(self):
        """更新夹爪控制（支持手动模式和恒力控制）"""
        gripper_actuators = _GRIP_ACT

        for side in ["left", "right"]:
            act_ids = gripper_actuators.get(side, [])
            if len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids):

                # ===== 优先级1：位置伺服锁定模式 =====
                if self.gripper_servo_lock[side] and self.gripper_lock_position[side] is not None:
                    # 位置伺服锁定：使用锁定的位置作为目标
                    # MuJoCo的position actuator会自动计算：force = kp × (target - current)
                    target_pos = self.gripper_lock_position[side]
                    for aid in act_ids:
                        lo, hi = model.actuator_ctrlrange[aid]
                        data.ctrl[aid] = max(lo, min(hi, target_pos))

                # ===== 优先级2：手动模式 =====
                elif self.gripper_manual_mode[side] and self.gripper_hold_positions[side] is not None:
                    # 手动模式：使用保持位置
                    target_pos = self.gripper_hold_positions[side]
                    for aid in act_ids:
                        lo, hi = model.actuator_ctrlrange[aid]
                        data.ctrl[aid] = max(lo, min(hi, target_pos))

                # ===== 优先级3：自动模式 =====
                else:
                    # 自动模式：使用控制器目标
                    target_pos = self.gripper_targets[side]
                    for aid in act_ids:
                        lo, hi = model.actuator_ctrlrange[aid]
                        data.ctrl[aid] = max(lo, min(hi, target_pos))

    def apply_gripper_force_lock(self):
        """在恒定夹紧力模式下向手指施加外力"""
        for side in ["left", "right"]:
            dofs = self.gripper_force_dofs[side]
            if self.gripper_force_lock[side] and self.gripper_force_target[side] > 0.0:
                force = -self.gripper_force_target[side]
                for dof in dofs:
                    if dof is not None and dof >= 0:
                        data.qfrc_applied[dof] = force
            else:
                for dof in dofs:
                    if dof is not None and dof >= 0:
                        data.qfrc_applied[dof] = 0.0
    
    def set_gripper_target(self, side, value):
        """设置夹爪目标位置（仅在非手动模式下生效）"""
        if self.gripper_manual_mode[side]:
            return  # 手动模式下忽略自动目标设置
            
        jnames = ["Joint_finger1","Joint_finger2"] if side=="left" else ["r_Joint_finger1","r_Joint_finger2"]
        for jn in jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                lo, hi = model.jnt_range[jid]
                self.gripper_targets[side] = max(lo, min(hi, value))
                break
    
    def set_14dim_target_qpos(self, mobile_aloha_qpos):
        """
        设置14维Mobile ALOHA格式的目标关节角度
        用于从Mobile ALOHA模型输出设置机器人目标状态
        """
        # 左臂6维
        for i in range(6):
            self.target_qpos["left"][i] = mobile_aloha_qpos[i]
        
        # 左夹爪1维 -> 分配给两个手指
        left_gripper_val = mobile_aloha_qpos[6]
        self.gripper_targets["left"] = left_gripper_val
        
        # 右臂6维
        for i in range(6):
            self.target_qpos["right"][i] = mobile_aloha_qpos[7 + i]
        
        # 右夹爪1维 -> 分配给两个手指
        right_gripper_val = mobile_aloha_qpos[13]
        self.gripper_targets["right"] = right_gripper_val
    
    def get_14dim_target_commands(self):
        """
        获取14维Mobile ALOHA格式的目标指令
        用于录制数据时区分state（当前状态）和command（目标指令）
        """
        result = np.zeros(14, dtype=np.float32)
        
        # 左臂6维目标
        for i in range(6):
            result[i] = self.target_qpos["left"][i]
        
        # 左夹爪1维目标
        if self.gripper_manual_mode["left"] and self.gripper_hold_positions["left"] is not None:
            # 手动模式：使用保持位置作为目标
            result[6] = self.gripper_hold_positions["left"]
        else:
            # 自动模式：使用控制器目标
            result[6] = self.gripper_targets["left"]
        
        # 右臂6维目标
        for i in range(6):
            result[7 + i] = self.target_qpos["right"][i]
        
        # 右夹爪1维目标
        if self.gripper_manual_mode["right"] and self.gripper_hold_positions["right"] is not None:
            # 手动模式：使用保持位置作为目标
            result[13] = self.gripper_hold_positions["right"]
        else:
            # 自动模式：使用控制器目标
            result[13] = self.gripper_targets["right"]
        
        return result

# ===== 改进的IK步进函数 =====
def ik_step_dynamic(controller, side, target_pos, target_quat=None, max_iters=5):
    """
    使用动力学兼容的方式进行IK控制。
    target_quat 提供时，使用6D IK保持末端姿态；否则退回到仅位置 IK。
    """
    if target_quat is not None:
        controller.set_target_from_ik_pose(side, target_pos, quat_normalize(target_quat))
    else:
        controller.set_target_from_ik(side, target_pos)
    
    # 多次物理步进以达到目标
    for _ in range(max_iters):
        controller.update_control()
        mujoco.mj_step(model, data)  # 使用物理步进而不是mj_forward

# ===== 夹爪控制函数 =====
def set_gripper(side, delta, controller):
    """
    同时写 qpos 和 actuator ctrl，立即可见且不会被拉回。
    delta>0 变大（张开），delta<0 变小（闭合）
    """
    # 启用手动模式
    controller.set_gripper_manual_mode(side, True)
    
    # 先算目标 qpos（两指一起动）
    jnames = ["Joint_finger1","Joint_finger2"] if side=="left" else ["r_Joint_finger1","r_Joint_finger2"]
    targets = []
    for jn in jnames:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        v = float(data.qpos[adr]) + float(delta)
        if lo < hi:
            v = max(lo, min(hi, v))
        data.qpos[adr] = v
        targets.append((adr, v))

    # 同步写 actuator ctrl（若存在）
    act_ids = _GRIP_ACT.get(side, [])
    if len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids):
        for aid in act_ids:
            lo, hi = model.actuator_ctrlrange[aid]
            # 用当前目标 v（两指同值即可）
            v = targets[0][1] if targets else data.ctrl[aid]
            v = max(lo, min(hi, v))
            data.ctrl[aid] = v

    # 更新控制器的保持位置
    controller.update_gripper_hold_position(side)
    
    mujoco.mj_forward(model, data)  # 立即刷新画面

def set_gripper_force_control(side, delta, controller, force_limit=10.0):
    """
    带力限制和碰撞检测的夹爪控制
    """
    # 检查当前是否有阻挡性碰撞
    contact_count = debug_gripper_contacts(side)
    
    if contact_count > 2 and delta < 0:  # 闭合时遇到较多碰撞
        print(f"[{side}] 检测到阻挡，停止闭合")
        return False
    
    # 正常控制
    controller.set_gripper_manual_mode(side, True)
    
    jnames = ["Joint_finger1","Joint_finger2"] if side=="left" else ["r_Joint_finger1","r_Joint_finger2"]
    targets = []
    
    for jn in jnames:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        
        current_pos = float(data.qpos[adr])
        new_pos = current_pos + float(delta)
        
        # 更保守的限制，为碰撞留余量
        if lo < hi:
            margin = 0.002  # 2mm的安全余量
            effective_lo = lo + margin
            effective_hi = hi - margin
            new_pos = max(effective_lo, min(effective_hi, new_pos))
        
        data.qpos[adr] = new_pos
        targets.append((adr, new_pos))

    # 设置执行器控制
    act_ids = _GRIP_ACT.get(side, [])
    if len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids):
        for aid in act_ids:
            lo, hi = model.actuator_ctrlrange[aid]
            v = targets[0][1] if targets else data.ctrl[aid]
            v = max(lo, min(hi, v))
            data.ctrl[aid] = v

    controller.update_gripper_hold_position(side)
    mujoco.mj_forward(model, data)
    return True

def print_gripper_state():
    def one(side):
        jnames = ["Joint_finger1","Joint_finger2"] if side=="left" else ["r_Joint_finger1","r_Joint_finger2"]
        q = []
        for jn in jnames:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                adr = model.jnt_qposadr[jid]
                q.append(float(data.qpos[adr]))
            except Exception:
                q.append(None)
        a = []
        for aid in _GRIP_ACT.get(side, []):
            a.append(float(data.ctrl[aid]) if (aid is not None and aid >= 0) else None)
        return q, a
    lq, la = one("left"); rq, ra = one("right")
    print(f"[gripper] L qpos={np.round(lq,4)} ctrl={np.round(la,4)} | R qpos={np.round(rq,4)} ctrl={np.round(ra,4)}")

def set_joint_delta(joint_name, delta, controller=None, chains=None):
    """
    增量式调一个关节：
      - 直接改 qpos 并夹到 joint <range>
      - 找到所有驱动该关节的 actuator，把 data.ctrl 同步为同一目标
    这样 Pause/Run 都能立刻看到效果
    """
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    except Exception:
        jid = -1
    if jid < 0:
        print(f"[WARN] joint not found: {joint_name}")
        return False

    # qpos += delta，并夹到 joint range
    adr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    v = float(data.qpos[adr]) + float(delta)
    if lo < hi:
        v = max(lo, min(hi, v))
    data.qpos[adr] = v

    # 同步所有"传动目标是这个关节"的 actuator 的 ctrl（通常是 position actuator）
    try:
        trn = model.actuator_trnid  # (nu, 2)
        for aid in range(model.nu):
            if trn[aid][0] == jid:  # 这个 actuator 驱动该关节
                clo, chi = model.actuator_ctrlrange[aid]
                tgt = max(clo, min(chi, v))
                data.ctrl[aid] = tgt
    except Exception:
        pass

    if controller is not None and chains is not None:
        try:
            for side in ("left", "right"):
                c = chains.get(side)
                if not c:
                    continue
                if jid in c["jids"]:
                    idx = c["jids"].index(jid)
                    controller.target_qpos[side][idx] = data.qpos[adr]
                    controller.integral_error[side][idx] = 0.0
                    break
        except Exception:
            pass

    mujoco.mj_forward(model, data)
    return True

def gripper_troubleshooting():
    """夹爪问题排查指南"""
    print("""
=== 夹爪无法闭合问题排查 ===

1. 检查碰撞：
   - 按 'c' 键调用 debug_gripper_contacts() 查看碰撞详情
   
2. 检查距离：
   - 按 'v' 键检查夹爪与方块距离
   
3. 检查关节限制：
   - 查看夹爪关节的 jnt_range 是否合理
   - 确认当前位置没有接近限制边界
   
4. 尝试不同控制模式：
   - 普通模式：z/x/n/m
   - 力控制模式：Z/X/N/M (大写，更保守的碰撞检测)
   
5. 物理参数调整建议：
   - 在XML中增加夹爪的 <motor> 力矩限制
   - 调整接触参数（contact stiffness, damping）
   - 考虑修改夹爪碰撞几何体大小

当前控制策略：
- 小写字母 (z/x/n/m): 普通控制，适合远离物体时
- 大写字母 (Z/X/N/M): 力控制，会检测碰撞并自动停止
""")

# ================== Recording (保持原有录制功能) ==================
class _VideoSink:
    def __init__(self, path, width, height, fps):
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.kind = None
        self._w = None
        self._warned = False
        try:
            import cv2  # type: ignore
            fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
            writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
            if writer is not None and writer.isOpened():
                self._w = writer
                self.kind = "cv2"
            else:
                if writer is not None:
                    writer.release()
        except Exception:
            self._w = None

        if self._w is None:
            try:
                import imageio  # type: ignore
                self._iio = imageio
                self._w = imageio.get_writer(self.path, fps=self.fps)
                self.kind = "iio"
            except Exception as e:
                print("[record] ERROR: no video backend (cv2 or imageio).", e)
                self.kind = None
                self._w = None

    def write(self, frame_rgb):
        if self._w is None:
            if not self._warned:
                print("[record] WARNING: video writer unavailable, skipping frames")
                self._warned = True
            return
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            try:
                import cv2  # type: ignore
                frame = cv2.resize(frame, (self.width, self.height))
            except Exception:
                frame = frame[:self.height, :self.width]
        if self.kind == "cv2":
            import cv2  # type: ignore
            if hasattr(self._w, "isOpened") and not self._w.isOpened():
                if not self._warned:
                    print("[record] WARNING: cv2 writer closed, stop writing")
                    self._warned = True
                return
            bgr = frame[..., ::-1].copy()
            self._w.write(bgr)
        elif self.kind == "iio":
            try:
                self._w.append_data(frame)
            except AttributeError:
                self._w.write(frame)

    def close(self):
        if self._w is None:
            return
        if self.kind == "cv2":
            self._w.release()
        elif self.kind == "iio":
            self._w.close()
        self._w = None

class Recorder:
    def __init__(self, model, data, cams=("top","left_wrist","right_wrist"), fps=15, size=(640,480), out_root=None):
        self.model = model
        self.data = data
        self.cams = list(cams)
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        if out_root is None:
            from pathlib import Path as _Path
            default_root = _Path(__file__).resolve().parent / "datasets"
        else:
            from pathlib import Path as _Path
            default_root = _Path(out_root)
        self.out_root = str(default_root)
        self.renderers = {}
        self.videos = {}
        self.csv = None
        self.csv_path = None
        self.meta = {}
        self.t0 = None
        self.last = 0.0
        self.interval = 1.0 / self.fps
        self.frame_id = 0
        self.enabled = False
        self.h5 = None
        self.h5_dsets = {}
        self.have_h5py = False
        
        # 添加action缓存
        self.action_buffer = []
        self.qpos_buffer = []
        self.qvel_buffer = []
        self.image_buffers = {cam: [] for cam in cams}
        self.combo_video = None
        self.composite_order = ("left_wrist", "right_wrist", "top")
        self._combo_warned = False
        self._plot_warned = False

    def _ensure_renderers(self):
        maxW = int(getattr(self.model.vis.global_, "offwidth", 320))
        maxH = int(getattr(self.model.vis.global_, "offheight", 240))
        reqW, reqH = int(self.size[0]), int(self.size[1])
        W = min(reqW, maxW) if maxW > 0 else reqW
        H = min(reqH, maxH) if maxH > 0 else reqH
        for name in self.cams:
            if name not in self.renderers:
                ok = False
                try:
                    self.renderers[name] = mujoco.Renderer(self.model, W, H)
                    ok = True
                    print(f"[record] renderer[{name}] size = {W}x{H} (offbuffer {maxW}x{maxH})")
                except Exception as e1:
                    try:
                        self.renderers[name] = mujoco.Renderer(self.model, H, W)
                        ok = True
                        print(f"[record] renderer[{name}] size = {H}x{W} (swapped; offbuffer {maxW}x{maxH})")
                    except Exception as e2:
                        print(f"[record] ERROR creating renderer for {name}: {e1} | swapped: {e2}")
                if not ok:
                    raise RuntimeError(f"Cannot create offscreen renderer for camera {name}")

    def start(self, chains, L_tgt, R_tgt):
        import json, os
        from pathlib import Path as _Path
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = _Path(self.out_root) / f"run_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)
        self._ensure_renderers()
        
        # 清空缓存
        self.action_buffer = []
        self.qpos_buffer = []
        self.qvel_buffer = []
        self.image_buffers = {cam: [] for cam in self.cams}
        
        # 视频录制（可选）
        for cam in self.cams:
            sink = _VideoSink(str(outdir / f"{cam}.mp4"), self.size[0], self.size[1], self.fps)
            self.videos[cam] = sink
        
        # 合成三路摄像头视频（左手-右手-顶部）
        if all(cam in self.cams for cam in self.composite_order):
            combo_name = "_".join(self.composite_order) + "_combo.mp4"
            self.combo_video = _VideoSink(str(outdir / combo_name), self.size[0] * len(self.composite_order), self.size[1], self.fps)
        else:
            self.combo_video = None
        self._combo_warned = False
        self._plot_warned = False
        
        # CSV录制（可选，用于调试）- 使用14维Mobile ALOHA格式
        self.csv_path = str(outdir / "states.csv")
        self.csv = open(self.csv_path, "w", newline="")
        import csv as _csv
        self.writer = _csv.writer(self.csv)
        
        # 14维Mobile ALOHA标准关节名称
        mobile_aloha_joint_names = [
            "l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6",  # 左臂6维
            "l_gripper",  # 左夹爪1维
            "r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6",  # 右臂6维  
            "r_gripper"   # 右夹爪1维
        ]
        header = (["t", "frame"] +
                  mobile_aloha_joint_names +
                  ["Lx","Ly","Lz","Lqw","Lqx","Lqy","Lqz",
                   "Rx","Ry","Rz","Rqw","Rqx","Rqy","Rqz"] +
                  ["L_tgt_x","L_tgt_y","L_tgt_z","R_tgt_x","R_tgt_y","R_tgt_z"])
        self.writer.writerow(header)
        
        # 元数据
        self.meta = {
            "fps": self.fps,
            "size": self.size,
            "cameras": self.cams,
            "original_nq": int(self.model.nq),  # CURI原始qpos维度
            "original_nv": int(self.model.nv),  # CURI原始qvel维度
            "mobile_aloha_qpos_dim": 14,        # Mobile ALOHA兼容qpos维度
            "mobile_aloha_format": "l_joint1-6, l_gripper, r_joint1-6, r_gripper",
            "gripper_mapping": "CURI双手指夹爪映射为单维夹爪 (取平均值)",
            "bodies": {
                "left_ee": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, chains["left"]["ee"]),
                "right_ee": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, chains["right"]["ee"]),
            },
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        
        # HDF5文件初始化
        try:
            import h5py
            self.have_h5py = True
            self.h5_path = str(outdir / 'episode_0.hdf5')  # ACT格式使用episode_X.hdf5
            # 暂不创建文件，等stop时一次性写入
        except Exception as e:
            self.have_h5py = False
            print('[record] h5py not available, will skip episode.hdf5:', e)

        self.outdir = str(outdir)
        self.t0 = time.time()
        self.last = 0.0
        self.frame_id = 0
        self.enabled = True
        import os as _os
        print(f"[record] START -> {_os.path.abspath(self.outdir)}  @ {self.fps} FPS  cams={self.cams}")

    def stop(self):
        if not self.enabled:
            return
            
        # 关闭视频文件
        for v in self.videos.values():
            try: v.close()
            except Exception: pass
        self.videos.clear()
        if self.combo_video is not None:
            try: self.combo_video.close()
            except Exception: pass
            self.combo_video = None
        self._combo_warned = False
        self._plot_warned = False
        
        # 关闭CSV文件
        if self.csv:
            self.csv.flush(); self.csv.close(); self.csv = None
        
        # 保存HDF5文件（ACT-Plus-Plus格式）
        if self.have_h5py and self.h5_path:
            try:
                import h5py
                
                # 创建HDF5文件
                with h5py.File(self.h5_path, 'w') as h5:
                    # 保存图像数据
                    # ACT格式通常使用 /observations/images/{camera_name}
                    obs_grp = h5.create_group('observations')
                    img_grp = obs_grp.create_group('images')
                    
                    # 摄像头名称映射（根据ACT的惯例）
                    cam_map = {
                        'top': 'top',
                        'left_wrist': 'left_wrist', 
                        'right_wrist': 'right_wrist'
                    }
                    
                    for cam_orig, cam_name in cam_map.items():
                        if cam_orig in self.image_buffers and len(self.image_buffers[cam_orig]) > 0:
                            img_data = np.array(self.image_buffers[cam_orig], dtype=np.uint8)
                            img_grp.create_dataset(cam_name, data=img_data, 
                                                  compression='gzip', compression_opts=4)
                            print(f"  - images/{cam_name}: {img_data.shape}")
                    
                    qpos_data = None
                    if len(self.qpos_buffer) > 0:
                        qpos_data = np.array(self.qpos_buffer, dtype=np.float32)
                        obs_grp.create_dataset('qpos', data=qpos_data,
                                              compression='gzip', compression_opts=4)
                        print(f"  - qpos: {qpos_data.shape}")
                    elif len(self.action_buffer) > 0:
                        qpos_data = np.array(self.action_buffer, dtype=np.float32)
                        obs_grp.create_dataset('qpos', data=qpos_data,
                                              compression='gzip', compression_opts=4)
                        print(f"  - qpos(fallback action): {qpos_data.shape}")

                    qvel_data = None
                    if len(self.qvel_buffer) > 0:
                        qvel_data = np.array(self.qvel_buffer, dtype=np.float32)
                        obs_grp.create_dataset('qvel', data=qvel_data,
                                              compression='gzip', compression_opts=4)
                        print(f"  - qvel: {qvel_data.shape}")

                    action_data = None
                    if len(self.action_buffer) > 0:
                        action_data = np.array(self.action_buffer, dtype=np.float32)
                        h5.create_dataset('action', data=action_data,
                                          compression='gzip', compression_opts=4)
                        print(f"  - action: {action_data.shape}")

                    # 兼容ALOHA可视化脚本，提供 root 下的 qpos/qvel 数据集
                    # 注意：action 已经在 root 级别创建（第1378行），无需重复创建
                    if qpos_data is not None and '/qpos' not in h5:
                        h5.create_dataset('/qpos', data=qpos_data,
                                          compression='gzip', compression_opts=4)
                    if qvel_data is not None and '/qvel' not in h5:
                        h5.create_dataset('/qvel', data=qvel_data,
                                          compression='gzip', compression_opts=4)
                    
                    # 保存元数据
                    h5.attrs['sim'] = True  # 标记这是仿真数据
                    h5.attrs['num_timesteps'] = len(self.qpos_buffer)
                    h5.attrs['fps'] = self.fps
                    
                import os as _os
                print(f"[record] episode_0.hdf5 saved at {_os.path.abspath(self.h5_path)}")
            except Exception as e:
                print(f"[record] Failed to save HDF5: {e}")
        
        import os as _os
        print(f"[record] STOP  -> {_os.path.abspath(self.outdir)}")
        self._save_qpos_plot()
        self.enabled = False

    def step(self, chains, L_tgt, R_tgt, controller=None):
        if not self.enabled or self.t0 is None:
            return
        t_rel = time.time() - self.t0
        if t_rel - self.last + 1e-9 < self.interval:
            return
        self.last += self.interval
        
        # 渲染并缓存图像
        frame_cache = {}
        for cam, rend in self.renderers.items():
            try:
                rend.update_scene(self.data, camera=cam)
                rgb = rend.render()
                frame_cache[cam] = rgb
                # 保存到单路视频
                if cam in self.videos:
                    self.videos[cam].write(rgb)
                # 缓存到内存（用于HDF5）
                self.image_buffers[cam].append(rgb.copy())
            except Exception as e:
                print(f"[record] camera {cam}: {e}")
        
        # 写入拼接视频
        if self.combo_video is not None:
            if all(cam in frame_cache for cam in self.composite_order):
                try:
                    combo = np.concatenate([frame_cache[cam] for cam in self.composite_order], axis=1)
                    self.combo_video.write(combo)
                except Exception as e:
                    if not self._combo_warned:
                        print(f"[record] combo video failed: {e}")
                        self._combo_warned = True
            elif not self._combo_warned:
                missing = [cam for cam in self.composite_order if cam not in frame_cache]
                print(f"[record] combo video skipped (missing frames: {missing})")
                self._combo_warned = True

        # 缓存状态数据 - 转换为14维Mobile ALOHA格式
        qpos_14dim = extract_14dim_qpos(self.data.qpos, chains)
        self.qpos_buffer.append(qpos_14dim)

        # 对qvel也进行相应的14维提取 (只提取对应的关节速度)
        qvel_14dim = np.zeros(14, dtype=np.float32)
        # 左臂6维速度
        left_chain = chains["left"]
        for i, jid in enumerate(left_chain["jids"]):
            dofadr = self.model.jnt_dofadr[jid]
            if dofadr >= 0:
                qvel_14dim[i] = self.data.qvel[dofadr]

        # 左夹爪1维速度 (平均)
        try:
            left_finger1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger1")
            left_finger2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger2")
            left_finger1_dofadr = self.model.jnt_dofadr[left_finger1_id]
            left_finger2_dofadr = self.model.jnt_dofadr[left_finger2_id]
            if left_finger1_dofadr >= 0 and left_finger2_dofadr >= 0:
                qvel_14dim[6] = (self.data.qvel[left_finger1_dofadr] + self.data.qvel[left_finger2_dofadr]) / 2.0
        except:
            qvel_14dim[6] = 0.0

        # 右臂6维速度
        right_chain = chains["right"]
        for i, jid in enumerate(right_chain["jids"]):
            dofadr = self.model.jnt_dofadr[jid]
            if dofadr >= 0:
                qvel_14dim[7 + i] = self.data.qvel[dofadr]

        # 右夹爪1维速度 (平均)
        try:
            right_finger1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger1")
            right_finger2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger2")
            right_finger1_dofadr = self.model.jnt_dofadr[right_finger1_id]
            right_finger2_dofadr = self.model.jnt_dofadr[right_finger2_id]
            if right_finger1_dofadr >= 0 and right_finger2_dofadr >= 0:
                qvel_14dim[13] = (self.data.qvel[right_finger1_dofadr] + self.data.qvel[right_finger2_dofadr]) / 2.0
        except:
            qvel_14dim[13] = 0.0

        self.qvel_buffer.append(qvel_14dim)

        # 对于action，使用控制器的目标指令而不是当前qpos
        if controller is not None:
            action_14dim = controller.get_14dim_target_commands()
        else:
            action_14dim = qpos_14dim.copy()

        self.action_buffer.append(action_14dim)

        # CSV记录（用于调试）- 使用14维Mobile ALOHA格式
        if self.csv and self.writer:
            Lp = self.data.xpos[chains["left"]["ee"]].copy()
            Lq = self.data.xquat[chains["left"]["ee"]].copy()
            Rp = self.data.xpos[chains["right"]["ee"]].copy()
            Rq = self.data.xquat[chains["right"]["ee"]].copy()

            row = [t_rel, self.frame_id] + qpos_14dim.tolist() + \
                  [*Lp.tolist(), *Lq.tolist(), *Rp.tolist(), *Rq.tolist(),
                   float(L_tgt[0]), float(L_tgt[1]), float(L_tgt[2]),
                   float(R_tgt[0]), float(R_tgt[1]), float(R_tgt[2])]
            try:
                self.writer.writerow(row)
            except Exception:
                pass

        self.frame_id += 1
    
    def _save_qpos_plot(self):
        if not self.qpos_buffer and self.action_buffer:
            self.qpos_buffer = [np.asarray(cmd, dtype=np.float32) for cmd in self.action_buffer]
        if not self.qpos_buffer:
            if not self._plot_warned:
                print("[record] skip qpos plot (no frames captured)")
                self._plot_warned = True
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            if not self._plot_warned:
                print(f"[record] matplotlib not available, skip qpos plot: {e}")
                self._plot_warned = True
            return
        try:
            data = np.array(self.qpos_buffer, dtype=np.float32)
            steps = data.shape[0]
            t = np.arange(steps)
            from pathlib import Path
            out_path = Path(self.outdir) / "episode_0_qpos.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            cmap = plt.get_cmap("tab10")

            left_names = ["L_joint0", "L_joint1", "L_joint2", "L_joint3", "L_joint4", "L_joint5", "L_gripper"]
            for i, (idx, name) in enumerate(zip(range(7), left_names)):
                axes[0].plot(t, data[:, idx], label=name, color=cmap(i % 10))
            axes[0].set_ylabel("Left arm / gripper (Mobile ALOHA)")
            axes[0].legend(loc="upper right", fontsize=8, ncol=3)
            axes[0].grid(True, linestyle="--", alpha=0.3)

            right_names = ["R_joint0", "R_joint1", "R_joint2", "R_joint3", "R_joint4", "R_joint5", "R_gripper"]
            for i, (idx, name) in enumerate(zip(range(7, 14), right_names)):
                axes[1].plot(t, data[:, idx], label=name, color=cmap(i % 10))
            axes[1].set_ylabel("Right arm / gripper (Mobile ALOHA)")
            axes[1].set_xlabel("Frame")
            axes[1].legend(loc="upper right", fontsize=8, ncol=3)
            axes[1].grid(True, linestyle="--", alpha=0.3)
            fig.suptitle("Episode qpos (Left 0-6, Right 7-13)")
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[record] qpos plot saved at {out_path}")
        except Exception as e:
            print(f"[record] Failed to save qpos plot: {e}")

# ================== Camera preview ==================
class CamPreview:
    def __init__(self, model, data, cams=("top","left_wrist","right_wrist"), size=(320,240), fps=5, window="cams"):
        self.model = model
        self.data = data
        self.cams = list(cams)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        self.interval = 1.0 / self.fps
        self.window = window
        self.enabled = False
        self.last_t = 0.0
        self.renderers = {}
        self._cv2 = None
        self._cv2_window_created = False
        self._qt_error_shown = False

        try:
            import cv2 as _cv2
            self._cv2 = _cv2
            # 启用Qt警告过滤
            _suppress_qt_warnings()
            # 测试cv2的GUI是否可用
            try:
                # 尝试创建一个测试窗口来检测Qt问题
                test_img = np.zeros((10, 10, 3), dtype=np.uint8)
                _cv2.imshow("__test__", test_img)
                _cv2.waitKey(1)
                _cv2.destroyWindow("__test__")
                print("[preview] OpenCV GUI 测试通过 (Qt警告已被抑制)")
            except Exception as e:
                print(f"[preview] 警告: OpenCV GUI不可用 (可能是WSL2/Qt问题): {e}")
                print("[preview] 摄像头预览功能将被禁用，但录制功能仍然可用")
                self._cv2 = None  # 禁用预览功能
        except Exception as e:
            print("[preview] OpenCV not available:", e, "-> run: pip install opencv-python")

        for name in self.cams:
            try:
                self.renderers[name] = mujoco.Renderer(model, self.size[0], self.size[1])
                print(f"[preview] renderer[{name}] size = {self.size[0]}x{self.size[1]}")
            except Exception as e:
                try:
                    self.renderers[name] = mujoco.Renderer(model, self.size[1], self.size[0])
                    print(f"[preview] renderer[{name}] size = {self.size[1]}x{self.size[0]} (swapped)")
                except Exception as e2:
                    print(f"[preview] ERROR creating renderer for {name}: {e} | swapped: {e2}")

    def toggle(self):
        if self._cv2 is None:
            if not self._qt_error_shown:
                print("[preview] 摄像头预览不可用 (OpenCV GUI在WSL2中存在问题)")
                print("[preview] 提示: 使用'R'键开始录制，录制功能不受影响")
                self._qt_error_shown = True
            return
        self.enabled = not self.enabled
        if not self.enabled:
            try:
                self._cv2.destroyWindow(self.window)
                self._cv2_window_created = False
            except Exception:
                pass
        print("[preview] enabled =", self.enabled)

    def step(self, t_now):
        if not self.enabled or self._cv2 is None:
            return
        if t_now - self.last_t + 1e-9 < self.interval:
            return
        self.last_t += self.interval

        try:
            frames = []
            for cam, rend in self.renderers.items():
                try:
                    rend.update_scene(self.data, camera=cam)
                    rgb = rend.render()
                    bgr = rgb[..., ::-1].copy()
                    self._cv2.putText(bgr, cam, (8, 20), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, self._cv2.LINE_AA)
                    frames.append(bgr)
                except Exception as e:
                    print(f"[preview] {cam}: {e}")
            if not frames:
                return
            try:
                panel = self._cv2.hconcat(frames)
            except Exception:
                H = min(f.shape[0] for f in frames); W = min(f.shape[1] for f in frames)
                frames = [f[:H, :W] for f in frames]
                panel = self._cv2.hconcat(frames)

            # 在主线程中安全地显示窗口
            self._cv2.imshow(self.window, panel)
            self._cv2_window_created = True
            self._cv2.waitKey(1)
        except Exception as e:
            # 捕获Qt线程错误，禁用预览功能
            if not self._qt_error_shown:
                print(f"[preview] GUI错误 (Qt线程问题): {e}")
                print("[preview] 自动禁用预览功能。录制功能不受影响。")
                self._qt_error_shown = True
            self.enabled = False
            self._cv2 = None

HELP = """
[Viewer+Terminal Controls] (keep TERMINAL focused)
  h    : help
  q    : quit
  0    : reset to keyframe 0
  [/]  : PID gains down/up
  -/=  : pos step down/up
  1/2/3: LEFT / RIGHT / BOTH
  g    : toggle go-to IK mode
  K    : toggle orientation lock (保持末端姿态)
  w/s, a/d, r/f : +Y/-Y, -X/+X, +Z/-Z (world)
  i/k, j/l, u/h : right arm +Y/-Y, -X/+X, +Z/-Z (world)
  
  === 夹爪控制 ===
  z/x  : 左夹爪闭合/张开 (普通模式)
  n/m  : 右夹爪闭合/张开 (普通模式)
  Z/X  : 左夹爪闭合/张开 (力控制模式，会检测碰撞)
  N/M  : 右夹爪闭合/张开 (力控制模式，会检测碰撞)

  === 恒定夹紧力模式 ===
  Ctrl+G : 切换左夹爪恒定夹紧力（默认10N）
  Ctrl+H : 切换右夹爪恒定夹紧力（默认10N）
  
  === 末端位姿控制 ===
  O    : 输入目标位姿 (位置 + 欧拉角)
  Q    : 输入目标位姿 (位置 + 四元数)
  E    : 紧急停止轨迹执行
  
  === 调试功能 ===
  c    : 检查夹爪碰撞状态
  v    : 检查夹爪与方块距离
  C    : 显示故障排查指南
  
  ,/.  : head_joint1 -/+
  ;/:  : head_joint2 -/+
  t/y  : platform_joint -/+
  R/S  : start/stop recording (3 cams)
  F    : toggle camera preview window
  P    : toggle physics (pause/resume)
  
[夹爪控制说明]
- 普通模式 (z/x/n/m): 适合远离物体时，步长较大
- 力控制模式 (Z/X/N/M): 适合靠近物体时，有碰撞检测，步长较小
- 恒定夹紧力 (Ctrl+G/Ctrl+H): 对当前夹持物体持续施加约10N的向内压力
  * 按下组合键后立即生效，再次按下即可解除
  * 适用场景：抓取后需要长期保持夹持力、防止物体滑落
- 松开按键后自动保持位置，不会回弹
- 按0重置时会释放所有夹爪控制模式
- 使用c键和v键来调试碰撞和距离问题
"""

def main():
    chains = find_arm_chain()
    print_chain(chains)

    # 创建控制器
    controller = TargetController(chains, kp=130.0, kd=28.0, ki=0.4)
    
    mujoco.mj_forward(model, data)
    # 随机设置红色方块位置
    try:
        box_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    except Exception:
        box_joint = -1
    if box_joint >= 0:
        adr = model.jnt_qposadr[box_joint]
        box_pos = np.array([
            np.random.uniform(-0.32, -0.28),
            np.random.uniform(-0.72, -0.68),
            0.57
        ], dtype=np.float64)
        box_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qpos[adr:adr+3] = box_pos
        data.qpos[adr+3:adr+7] = box_quat
        mujoco.mj_forward(model, data)
        info = {"timestamp": time.time(), "box_pos": box_pos.tolist(), "box_quat": box_quat.tolist()}
        automation_dir = Path(__file__).resolve().parent / "automation"
        automation_dir.mkdir(parents=True, exist_ok=True)
        (automation_dir / "box_pose.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"[init] red_box placed at {np.round(box_pos,4)}")

    L_tgt = data.xpos[chains["left"]["ee"]].copy()
    R_tgt = data.xpos[chains["right"]["ee"]].copy()
    L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
    R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()

    mode = "left"
    step = 0.001
    goto = True
    physics_paused = False
    keep_orientation = True  # True: 按键平移时保持末端姿态

    # ===== 新增：轨迹执行相关变量 =====
    trajectory_active = False
    trajectory_side = None
    trajectory_positions = []
    trajectory_quaternions = []
    trajectory_index = 0
    trajectory_start_time = None
    trajectory_duration = 0.0
    
    preview = CamPreview(model, data, cams=("top","left_wrist","right_wrist"), size=(320,240), fps=15)
    record_root = "/mnt/d/mycodes/act++/curi1/datasets/sim_transfer_cube_scripted_curi1"
    recorder = Recorder(
        model,
        data,
        cams=("top","left_wrist","right_wrist"),
        fps=60,
        size=(640,480),
        out_root=record_root,
    )

    # ===== 新增：轨迹生成函数 =====
    def generate_trajectory(side, target_pos, target_quat, seconds=1.0, fps=15):
        """生成平滑的轨迹点"""
        seconds = max(0.01, seconds * TRAJECTORY_TIME_SCALE)
        steps = max(1, int(seconds * fps))
        c = controller.chains[side]
        p0 = data.xpos[c["ee"]].copy()
        q0 = data.xquat[c["ee"]].copy()
        
        positions = []
        quaternions = []
        
        # 采用加速-匀速-减速（trapezoid）速度曲线，平滑控制长距离运动
        accel_ratio = 0.001
        decel_ratio = 0.001
        if accel_ratio + decel_ratio > 0.9:
            scale = 0.9 / (accel_ratio + decel_ratio)
            accel_ratio *= scale
            decel_ratio *= scale
        const_ratio = max(0.0, 1.0 - accel_ratio - decel_ratio)
        # 计算速度/加速度参数，使总位移正好为1
        vmax = 1.0 / (const_ratio + 0.5 * (accel_ratio + decel_ratio))
        acc = vmax / accel_ratio if accel_ratio > 1e-8 else 0.0
        dec = vmax / decel_ratio if decel_ratio > 1e-8 else 0.0
        s_acc = 0.5 * vmax * accel_ratio
        s_const = vmax * const_ratio

        for i in range(steps):
            tau = (i + 1) / steps
            if accel_ratio > 0 and tau < accel_ratio:
                s = 0.5 * acc * tau**2
            elif tau < accel_ratio + const_ratio or decel_ratio == 0:
                s = s_acc + vmax * (tau - accel_ratio)
            else:
                t_dec = tau - (accel_ratio + const_ratio)
                s = s_acc + s_const + vmax * t_dec - 0.5 * dec * (t_dec**2)
            s = np.clip(s, 0.0, 1.0)

            p = (1.0 - s) * p0 + s * target_pos
            q = quat_slerp(q0, target_quat, s)

            positions.append(p.copy())
            quaternions.append(q.copy())

        return positions, quaternions

    def start_pose_trajectory(side, target_pos, target_quat, duration, fps=10.0):
        nonlocal trajectory_active, trajectory_side, trajectory_positions, trajectory_quaternions
        nonlocal trajectory_index, trajectory_start_time, trajectory_duration
        duration_scaled = float(duration) * TRAJECTORY_TIME_SCALE
        traj = generate_trajectory(side, target_pos, target_quat, seconds=duration_scaled, fps=float(fps))
        positions, quats = traj
        if len(positions) == 0:
            return None
        trajectory_positions = positions
        trajectory_quaternions = quats
        trajectory_active = True
        trajectory_side = side
        trajectory_index = 0
        trajectory_start_time = time.time()
        trajectory_duration = duration_scaled
        return quats[-1].copy()

    automation_queue = []
    automation_current = None
    automation_wait_until = None
    automation_dir = Path(__file__).resolve().parent / "automation"
    automation_dir.mkdir(parents=True, exist_ok=True)
    automation_file = automation_dir / "commands.json"

    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(3)
        import sys, os, select, termios, tty
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
        print("\n[INFO] 现在使用动力学兼容的控制方式")
        print("[INFO] 碰撞检测已启用，物理模拟正常运行")
        print("[INFO] 夹爪保持功能已完全集成到控制器中")
        print("[INFO] 按O/Q键输入目标位姿，将看到连续平滑的运动")
        
        with raw_terminal_mode(sys.stdin):
            while viewer.is_running():
                for ch in read_keys(0.0):
                    if ch in ('B'):
                        print(HELP)
                    elif ch == 'q':
                        return
                    elif ch == '0':
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                        data.qvel[:] = 0; data.act[:] = 0
                        mujoco.mj_forward(model, data)
                        L_tgt = data.xpos[chains["left"]["ee"]].copy()
                        R_tgt = data.xpos[chains["right"]["ee"]].copy()
                        L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                        R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                        # 重置时释放所有夹爪控制模式
                        controller.set_gripper_manual_mode("left", False)
                        controller.set_gripper_manual_mode("right", False)
                        controller.gripper_servo_lock["left"] = False
                        controller.gripper_servo_lock["right"] = False
                        controller.gripper_lock_position["left"] = None
                        controller.gripper_lock_position["right"] = None
                        controller.gripper_force_lock["left"] = False
                        controller.gripper_force_lock["right"] = False
                        controller.gripper_force_target["left"] = 0.0
                        controller.gripper_force_target["right"] = 0.0
                        print("[reset] keyframe 0, all gripper modes released")
                    elif ch == '[':
                        controller.kp *= 0.8; controller.kd *= 0.8
                        print(f"[gains] kp={controller.kp:.1f} kd={controller.kd:.1f}")
                    elif ch == ']':
                        controller.kp *= 1.25; controller.kd *= 1.25
                        print(f"[gains] kp={controller.kp:.1f} kd={controller.kd:.1f}")
                    elif ch == '-':
                        step = max(0.001, step/1.5); print(f"[step] -> {step:.4f} m")
                    elif ch in ('=','+'):
                        step = min(0.10, step*1.5); print(f"[step] -> {step:.4f} m")
                    elif ch == '1':
                        mode = "left"; print("[mode] LEFT")
                    elif ch == '2':
                        mode = "right"; print("[mode] RIGHT")
                    elif ch == '3':
                        mode = "both"; print("[mode] BOTH")
                    elif ch == 'g':
                        goto = not goto; print(f"[goto] {'ON' if goto else 'OFF'}")
                    elif ch == 'K':
                        keep_orientation = not keep_orientation
                        print(f"[orientation] 姿态保持 {'ON' if keep_orientation else 'OFF'}")
                    elif ch == 's':
                        if mode in ('left','both'): L_tgt[1] -= step
                    elif ch == 'w':
                        if mode in ('left','both'): L_tgt[1] += step
                    elif ch == 'a':
                        if mode in ('left','both'): L_tgt[0] -= step
                    elif ch == 'd':
                        if mode in ('left','both'): L_tgt[0] += step
                    elif ch == 'r':
                        if mode in ('left','both'): L_tgt[2] += step
                    elif ch == 'f':
                        if mode in ('left','both'): L_tgt[2] -= step
                    elif ch == 'k':
                        if mode in ('right','both'): R_tgt[1] -= step
                    elif ch == 'i':
                        if mode in ('right','both'): R_tgt[1] += step
                    elif ch == 'j':
                        if mode in ('right','both'): R_tgt[0] -= step
                    elif ch == 'l':
                        if mode in ('right','both'): R_tgt[0] += step
                    elif ch == 'u':
                        if mode in ('right','both'): R_tgt[2] += step
                    elif ch == 'h':
                        if mode in ('right','both'): R_tgt[2] -= step
                    # Head & platform joints
                    elif ch == ',': set_joint_delta('head_joint1', -0.02)
                    elif ch == '.': set_joint_delta('head_joint1', +0.02)
                    elif ch == ';': set_joint_delta('head_joint2', -0.02)
                    elif ch == ':': set_joint_delta('head_joint2', +0.02)
                    elif ch == '4':
                        if set_joint_delta('l_joint4', -0.02, controller, chains):
                            L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                    elif ch == '$':
                        if set_joint_delta('l_joint4', +0.02, controller, chains):
                            L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                    elif ch == '5':
                        if set_joint_delta('l_joint5', -0.02, controller, chains):
                            L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                    elif ch == '%':
                        if set_joint_delta('l_joint5', +0.02, controller, chains):
                            L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                    elif ch == '6':
                        if set_joint_delta('l_joint6', -0.02, controller, chains):
                            L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                    elif ch == '^':
                        if set_joint_delta('l_joint6', +0.02, controller, chains):
                            L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                    elif ch == '7':
                        if set_joint_delta('r_joint4', -0.02, controller, chains):
                            R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                    elif ch == '&':
                        if set_joint_delta('r_joint4', +0.02, controller, chains):
                            R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                    elif ch == '8':
                        if set_joint_delta('r_joint5', -0.02, controller, chains):
                            R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                    elif ch == '*':
                        if set_joint_delta('r_joint5', +0.02, controller, chains):
                            R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                    elif ch == '9':
                        if set_joint_delta('r_joint6', -0.02, controller, chains):
                            R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                    elif ch == '(':
                        if set_joint_delta('r_joint6', +0.02, controller, chains):
                            R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                    elif ch == 't': 
                        _pL0 = data.xpos[chains['left']['base']].copy()
                        _pR0 = data.xpos[chains['right']['base']].copy()
                        set_joint_delta('platform_joint', -0.01)
                        _pL1 = data.xpos[chains['left']['base']].copy()
                        _pR1 = data.xpos[chains['right']['base']].copy()
                        L_tgt += (_pL1 - _pL0)
                        R_tgt += (_pR1 - _pR0)
                    elif ch == 'y':                
                        _pL0 = data.xpos[chains['left']['base']].copy()
                        _pR0 = data.xpos[chains['right']['base']].copy()
                        set_joint_delta('platform_joint', +0.01)
                        _pL1 = data.xpos[chains['left']['base']].copy()
                        _pR1 = data.xpos[chains['right']['base']].copy()
                        L_tgt += (_pL1 - _pL0)
                        R_tgt += (_pR1 - _pR0)
                    # ===== 夹爪控制（普通模式）=====
                    elif ch == 'z': 
                        set_gripper("left", -0.001, controller)
                        print_gripper_state()
                        print("[gripper] Left manual mode ON (normal)")
                    elif ch == 'x': 
                        set_gripper("left", +0.001, controller)
                        print_gripper_state()
                        print("[gripper] Left manual mode ON (normal)")
                    elif ch == 'n': 
                        set_gripper("right", -0.001, controller)
                        print_gripper_state()
                        print("[gripper] Right manual mode ON (normal)")
                    elif ch == 'm': 
                        set_gripper("right", +0.001, controller)
                        print_gripper_state()
                        print("[gripper] Right manual mode ON (normal)")
                    
                    # ===== 夹爪控制（力控制模式）=====
                    elif ch == 'Z': 
                        is_close, dist = check_cube_proximity("left")
                        if is_close and dist < 0.08:  # 8cm内使用小步长和碰撞检测
                            success = set_gripper_force_control("left", -0.002, controller)
                            if not success:
                                print("[left] 夹爪遇到阻力，已停止")
                        else:
                            set_gripper("left", -0.003, controller)
                        print_enhanced_gripper_state()
                        print("[gripper] Left manual mode ON (force control)")
                    elif ch == 'X': 
                        set_gripper("left", +0.003, controller)
                        print_enhanced_gripper_state()
                        print("[gripper] Left manual mode ON (force control)")
                    elif ch == 'N': 
                        is_close, dist = check_cube_proximity("right")
                        if is_close and dist < 0.08:  # 8cm内使用小步长和碰撞检测
                            success = set_gripper_force_control("right", -0.002, controller)
                            if not success:
                                print("[right] 夹爪遇到阻力，已停止")
                        else:
                            set_gripper("right", -0.003, controller)
                        print_enhanced_gripper_state()
                        print("[gripper] Right manual mode ON (force control)")
                    elif ch == 'M': 
                        set_gripper("right", +0.003, controller)
                        print_enhanced_gripper_state()
                        print("[gripper] Right manual mode ON (force control)")
                    
                    # ===== 调试功能 =====
                    # ===== 恒定夹紧力模式（Ctrl+G / Ctrl+H）=====
                    elif ch == '\x07':  # Ctrl+G (ASCII 7)
                        locked = controller.toggle_gripper_force_lock("left", force_newton=10.0)
                        if locked:
                            print("[info] 左夹爪恒力锁定已启用 (10N)")
                        else:
                            print("[info] 左夹爪恒力锁定已解除")
                        print_gripper_state()

                    elif ch == '\x08':  # Ctrl+H (ASCII 8)
                        locked = controller.toggle_gripper_force_lock("right", force_newton=10.0)
                        if locked:
                            print("[info] 右夹爪恒力锁定已启用 (10N)")
                        else:
                            print("[info] 右夹爪恒力锁定已解除")
                        print_gripper_state()

                    elif ch == 'c':
                        debug_gripper_contacts("left")
                        debug_gripper_contacts("right")
                    elif ch == 'v':
                        check_cube_proximity("left")
                        check_cube_proximity("right")
                    elif ch == 'C':
                        gripper_troubleshooting()
                    
                    # ===== 修改后的Pose IK触发 =====
                    elif ch == 'O':  # 输入位置 + 欧拉角(度)
                        try:
                            line = prompt_line("格式: side x y z roll pitch yaw(deg) duration(s)\n例如: right 0.20 -0.60 0.65 0 0 0 2\n> ")
                            parts = line.strip().split()
                            if len(parts) != 8:
                                print("输入格式错误，需要8个参数")
                                continue
                                
                            side, x, y, z, r, p_, yw, dur = parts
                            pos  = np.array([float(x), float(y), float(z)], dtype=np.float64)
                            quat = rpy_to_quat(np.deg2rad(float(r)), np.deg2rad(float(p_)), np.deg2rad(float(yw)))
                            final_quat = start_pose_trajectory(side.lower(), pos, quat, duration=float(dur), fps=60)
                            if final_quat is not None:
                                if side.lower() == "left":
                                    L_ori_tgt = final_quat
                                else:
                                    R_ori_tgt = final_quat
                                print(f"[POSE-IK] 开始执行轨迹: {side} -> pos={pos}, rpy=({r},{p_},{yw})deg, 持续{dur}秒")
                            else:
                                print("无法生成轨迹，请检查目标位置")
                            
                        except Exception as e:
                            print("输入解析失败:", e)

                    elif ch == 'Q':  # 输入位置 + 四元数
                        try:
                            line = prompt_line("格式: side x y z qw qx qy qz duration(s)\n例如: left 0.15 -0.55 0.70 1 0 0 0 1.5\n> ")
                            parts = line.strip().split()
                            if len(parts) != 9:
                                print("输入格式错误，需要9个参数")
                                continue
                                
                            side, x, y, z, qw, qx, qy, qz, dur = parts
                            pos  = np.array([float(x), float(y), float(z)], dtype=np.float64)
                            quat = quat_normalize(np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float64))
                            final_quat = start_pose_trajectory(side.lower(), pos, quat, duration=float(dur), fps=60)
                            if final_quat is not None:
                                if side.lower() == "left":
                                    L_ori_tgt = final_quat
                                else:
                                    R_ori_tgt = final_quat
                                print(f"[POSE-IK] 开始执行轨迹: {side} -> pos={pos}, quat={quat}, 持续{dur}秒")
                            else:
                                print("无法生成轨迹，请检查目标位置")
                            
                        except Exception as e:
                            print("输入解析失败:", e)

                    elif ch == 'E':  # Emergency stop - 紧急停止当前轨迹
                        if trajectory_active:
                            trajectory_active = False
                            if trajectory_side == "left":
                                L_ori_tgt = data.xquat[chains["left"]["ee"]].copy()
                            elif trajectory_side == "right":
                                R_ori_tgt = data.xquat[chains["right"]["ee"]].copy()
                            if automation_current and automation_current.get("type") == "move_pose":
                                automation_current = None
                                print("[automation] move cancelled")
                            print("[STOP] 轨迹执行已停止")
                    
                    # Head & platform joints
                    elif ch == ',': set_joint_delta('head_joint1', -0.02)
                    elif ch == '.': set_joint_delta('head_joint1', +0.02)
                    elif ch == ';': set_joint_delta('head_joint2', -0.02)
                    elif ch == ':': set_joint_delta('head_joint2', +0.02)
                    elif ch == 't': 
                        _pL0 = data.xpos[chains['left']['base']].copy()
                        _pR0 = data.xpos[chains['right']['base']].copy()
                        set_joint_delta('platform_joint', -0.01)
                        _pL1 = data.xpos[chains['left']['base']].copy()
                        _pR1 = data.xpos[chains['right']['base']].copy()
                        L_tgt += (_pL1 - _pL0)
                        R_tgt += (_pR1 - _pR0)
                    elif ch == 'y':                
                        _pL0 = data.xpos[chains['left']['base']].copy()
                        _pR0 = data.xpos[chains['right']['base']].copy()
                        set_joint_delta('platform_joint', +0.01)
                        _pL1 = data.xpos[chains['left']['base']].copy()
                        _pR1 = data.xpos[chains['right']['base']].copy()
                        L_tgt += (_pL1 - _pL0)
                        R_tgt += (_pR1 - _pR0)
                    elif ch == 'R':
                        try:
                            recorder.start(chains, L_tgt, R_tgt)
                        except Exception as e:
                            print('[record] start failed:', e)
                    elif ch == 'S':
                        try:
                            recorder.stop()
                        except Exception as e:
                            print('[record] stop failed:', e)
                    elif ch == 'F':
                        try:
                            preview.toggle()
                        except Exception as e:
                            print('[preview] toggle failed:', e)
                    elif ch == 'P':
                        physics_paused = not physics_paused
                        print(f"[physics] {'PAUSED' if physics_paused else 'RUNNING'}")
                    elif ch == 'b':  # 打印左右手末端 位置 + 四元数 + 欧拉角
                        print("Print the dual hand pos quat rpy_angles:")
                        print_ee_pose(controller)
                        # 两侧一起
                        # 如果只想打印单侧，也可以用：
                        # print_ee_pose(controller, 'left')
                        # print_ee_pose(controller, 'right')

                # ===== 自动化命令处理 =====
                if automation_file.exists():
                    try:
                        with open(automation_file, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        if isinstance(payload, dict):
                            commands = payload.get("commands", [])
                        elif isinstance(payload, list):
                            commands = payload
                        else:
                            commands = []
                        if commands:
                            automation_queue.extend(commands)
                            print(f"[automation] queued {len(commands)} commands")
                    except Exception as e:
                        print(f"[automation] failed to load commands: {e}")
                    finally:
                        try:
                            automation_file.unlink()
                        except OSError:
                            pass

                if automation_current and automation_current.get("type") == "sleep":
                    if automation_wait_until is not None and time.time() >= automation_wait_until:
                        print("[automation] sleep completed")
                        automation_current = None
                        automation_wait_until = None

                while automation_current is None and automation_queue:
                    cmd_peek = automation_queue[0]
                    ctype = (cmd_peek.get("type") or "").lower()
                    if ctype in {"record", "record_start", "record_stop"}:
                        cmd = automation_queue.pop(0)
                        action = (cmd.get("action") or ctype.split("_")[-1]).lower()
                        try:
                            if action in {"start", "on"}:
                                if not recorder.enabled:
                                    recorder.start(chains, L_tgt, R_tgt)
                                    print("[automation] recording START")
                                else:
                                    print("[automation] recording already running")
                            elif action in {"stop", "off"}:
                                if recorder.enabled:
                                    recorder.stop()
                                    print("[automation] recording STOP")
                                else:
                                    print("[automation] recording already stopped")
                            else:
                                print(f"[automation] unknown record action: {action}")
                        except Exception as e:
                            print(f"[automation] record failed: {e}")
                        continue  # 继续处理队列中的下一条
                    elif ctype == "gripper":
                        cmd = automation_queue.pop(0)
                        side = cmd.get("side", "right").lower()
                        target = cmd.get("target", "open")
                        value = cmd.get("value")
                        try:
                            if isinstance(target, str):
                                t_lower = target.lower()
                                if t_lower == "open":
                                    # 支持渐进张开
                                    duration = float(cmd.get("duration", 0.0))  # 总时长（秒）
                                    step_size = float(cmd.get("step_size", 0.001))  # 每步步长（米）
                                    step_sleep = float(cmd.get("step_sleep", 0.1))  # 每步间隔（秒）

                                    if duration > 0 or step_size < 0.03:
                                        # 渐进模式：计算当前位置，生成多步指令
                                        jnames = ["Joint_finger1","Joint_finger2"] if side == "left" else ["r_Joint_finger1","r_Joint_finger2"]
                                        readings = []
                                        for jn in jnames:
                                            try:
                                                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                                            except Exception:
                                                jid = -1
                                            if jid < 0:
                                                continue
                                            adr = model.jnt_qposadr[jid]
                                            readings.append(float(data.qpos[adr]))

                                        if readings:
                                            current = float(np.mean(readings))
                                            target_pos = current + 0.04  # 张开到最大（+4cm）

                                            # 根据 duration 或 step_size 计算步数
                                            if duration > 0:
                                                steps = max(1, int(duration / step_sleep))
                                            else:
                                                steps = max(1, int(0.04 / step_size))

                                            intermediate = np.linspace(current, target_pos, steps + 1)[1:]
                                            new_cmds = []
                                            for val in intermediate:
                                                new_cmds.append({
                                                    "type": "gripper",
                                                    "side": side,
                                                    "target": float(val),
                                                    "split": True,
                                                })
                                                if step_sleep > 0:
                                                    new_cmds.append({"type": "sleep", "seconds": step_sleep})
                                            automation_queue = new_cmds + automation_queue
                                            print(f"[automation] gripper {side} -> open (gradual, {steps} steps)")
                                            continue

                                    # 瞬间模式（默认，保持向后兼容）
                                    set_gripper(side, +0.0005, controller)
                                    print(f"[automation] gripper {side} -> open (instant)")

                                elif t_lower == "close":
                                    # 支持渐进闭合
                                    duration = float(cmd.get("duration", 0.0))
                                    step_size = float(cmd.get("step_size", 0.005))
                                    step_sleep = float(cmd.get("step_sleep", 0.1))
                                    delta = float(cmd.get("delta", 0.001))

                                    if duration > 0 or step_size < abs(delta):
                                        # 渐进模式
                                        jnames = ["Joint_finger1","Joint_finger2"] if side == "left" else ["r_Joint_finger1","r_Joint_finger2"]
                                        readings = []
                                        for jn in jnames:
                                            try:
                                                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                                            except Exception:
                                                jid = -1
                                            if jid < 0:
                                                continue
                                            adr = model.jnt_qposadr[jid]
                                            readings.append(float(data.qpos[adr]))

                                        if readings:
                                            current = float(np.mean(readings))
                                            target_pos = max(0.0, current - abs(delta))

                                            if duration > 0:
                                                steps = max(1, int(duration / step_sleep))
                                            else:
                                                steps = max(1, int(abs(delta) / step_size))

                                            intermediate = np.linspace(current, target_pos, steps + 1)[1:]
                                            new_cmds = []
                                            for val in intermediate:
                                                new_cmds.append({
                                                    "type": "gripper",
                                                    "side": side,
                                                    "target": float(val),
                                                    "split": True,
                                                })
                                                if step_sleep > 0:
                                                    new_cmds.append({"type": "sleep", "seconds": step_sleep})
                                            automation_queue = new_cmds + automation_queue
                                            print(f"[automation] gripper {side} -> close (gradual, {steps} steps)")
                                            continue

                                    # 瞬间模式
                                    set_gripper(side, -abs(delta), controller)
                                    print(f"[automation] gripper {side} -> close (instant)")

                                elif t_lower in {"stop", "hold", "hold_position"}:
                                    controller.set_gripper_manual_mode(side, True)
                                    controller.update_gripper_hold_position(side)
                                    print(f"[automation] gripper {side} -> {target}")
                                else:
                                    print(f"[automation] unknown gripper target: {target}")
                                    continue
                            else:
                                try:
                                    numeric = float(target)
                                except Exception:
                                    numeric = None
                                if numeric is None and value is not None:
                                    numeric = float(value)
                                if numeric is None:
                                    print(f"[automation] gripper target invalid: {target}")
                                    continue
                                # Determine current opening (meters)
                                jnames = ["Joint_finger1","Joint_finger2"] if side == "left" else ["r_Joint_finger1","r_Joint_finger2"]
                                readings = []
                                for jn in jnames:
                                    try:
                                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                                    except Exception:
                                        jid = -1
                                    if jid < 0:
                                        continue
                                    adr = model.jnt_qposadr[jid]
                                    readings.append(float(data.qpos[adr]))
                                if readings:
                                    current = float(np.mean(readings))
                                else:
                                    current = 0.0
                                diff = numeric - current
                                if abs(diff) < 1e-4:
                                    print(f"[automation] gripper {side} already at target {numeric:.3f}")
                                    continue
                                step_size = float(cmd.get("step_size", 0.005))
                                step_sleep = float(cmd.get("step_sleep", 0.2))
                                split = bool(cmd.get("split", False))
                                if not split:
                                    step_size = max(step_size, 1e-4)
                                    steps = max(1, int(abs(diff) / step_size))
                                    if steps > 1:
                                        intermediate = np.linspace(current, numeric, steps + 1)[1:]
                                        new_cmds = []
                                        for val in intermediate:
                                            new_cmds.append({
                                                "type": "gripper",
                                                "side": side,
                                                "target": float(val),
                                                "split": True,
                                                "step_size": step_size,
                                                "step_sleep": step_sleep,
                                            })
                                            if step_sleep > 0:
                                                new_cmds.append({"type": "sleep", "seconds": step_sleep})
                                        automation_queue = new_cmds + automation_queue
                                        continue
                                controller.set_gripper_manual_mode(side, True)
                                jnames = ["Joint_finger1","Joint_finger2"] if side == "left" else ["r_Joint_finger1","r_Joint_finger2"]
                                for jn in jnames:
                                    try:
                                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                                    except Exception:
                                        jid = -1
                                    if jid < 0:
                                        continue
                                    adr = model.jnt_qposadr[jid]
                                    lo, hi = model.jnt_range[jid]
                                    val = max(lo, min(hi, numeric))
                                    data.qpos[adr] = val
                                controller.update_gripper_hold_position(side)
                                mujoco.mj_forward(model, data)
                                print(f"[automation] gripper {side} -> {numeric}")
                        except Exception as e:
                            print(f"[automation] gripper failed: {e}")
                        continue
                    elif ctype == "move_pose":
                        if trajectory_active:
                            break  # 等待当前轨迹完成
                        cmd = automation_queue.pop(0)
                        side = cmd.get("side", "left").lower()
                        try:
                            pos = np.array(cmd["pos"], dtype=np.float64)
                        except Exception:
                            pos = data.xpos[chains[side]["ee"]].copy()
                        if "quat" in cmd:
                            quat = quat_normalize(np.array(cmd["quat"], dtype=np.float64))
                        elif "rpy_deg" in cmd:
                            rpy = np.deg2rad(np.array(cmd["rpy_deg"], dtype=np.float64))
                            quat = rpy_to_quat(*rpy)
                        elif "rpy_rad" in cmd:
                            quat = rpy_to_quat(*cmd["rpy_rad"])
                        else:
                            quat = data.xquat[chains[side]["ee"]].copy()
                        duration = float(cmd.get("duration", 2.0))
                        fps_cmd = float(cmd.get("fps", 60.0))
                        final_quat = start_pose_trajectory(side, pos, quat, duration=duration, fps=fps_cmd)
                        if final_quat is None:
                            print(f"[automation] move_pose failed (side={side})")
                            continue
                        if side == "left":
                            L_ori_tgt = final_quat
                        else:
                            R_ori_tgt = final_quat
                        automation_current = cmd
                        effective_duration = duration * TRAJECTORY_TIME_SCALE
                        print(f"[automation] move_pose -> {side} pos={np.round(pos,4)} duration={effective_duration:.2f}s (scaled)")
                        break
                    elif ctype == "sleep":
                        cmd = automation_queue.pop(0)
                        seconds = max(0.0, float(cmd.get("seconds", 0.0)))
                        automation_wait_until = time.time() + seconds
                        automation_current = cmd
                        print(f"[automation] sleep {seconds:.2f}s")
                        break
                    elif ctype == "gripper_force_lock":
                        cmd = automation_queue.pop(0)
                        side = cmd.get("side", "right").lower()
                        mode = cmd.get("mode", "toggle").lower()
                        force = float(cmd.get("force", 10.0))
                        if side not in ("left", "right"):
                            print(f"[automation] invalid gripper side: {side}")
                            continue
                        if mode == "enable":
                            if not controller.gripper_force_lock[side]:
                                controller.toggle_gripper_force_lock(side, force_newton=force)
                            else:
                                controller.gripper_force_target[side] = abs(force)
                            print(f"[automation] gripper_force_lock {side} ENABLE ({force:.2f}N)")
                        elif mode == "disable":
                            if controller.gripper_force_lock[side]:
                                controller.toggle_gripper_force_lock(side, force_newton=force)
                            print(f"[automation] gripper_force_lock {side} DISABLE")
                        else:  # toggle
                            locked = controller.toggle_gripper_force_lock(side, force_newton=force)
                            state = "ENABLE" if locked else "DISABLE"
                            print(f"[automation] gripper_force_lock {side} {state} ({force:.2f}N)")
                        # 该命令为即时执行，继续处理队列中的下一条
                        continue
                    else:
                        cmd = automation_queue.pop(0)
                        print(f"[automation] unknown command type: {ctype}")

                # ===== 轨迹执行逻辑 =====
                if trajectory_active and not physics_paused:
                    if trajectory_index < len(trajectory_positions):
                        # 获取当前目标点
                        target_pos = trajectory_positions[trajectory_index]
                        target_quat = trajectory_quaternions[trajectory_index]
                        
                        # 使用IK计算目标关节角度
                        controller.set_target_from_ik_pose(
                            trajectory_side, target_pos, target_quat, 
                            iters=10, damping=1e-4
                        )
                        if trajectory_side == "left":
                            L_ori_tgt = target_quat.copy()
                        else:
                            R_ori_tgt = target_quat.copy()
                        
                        # 移动到下一个轨迹点
                        trajectory_index += 1
                        
                        # 显示进度
                        if trajectory_index % 10 == 0:  # 每10个点显示一次
                            progress = (trajectory_index / len(trajectory_positions)) * 100
                            current_pos = data.xpos[controller.chains[trajectory_side]["ee"]].copy()
                            distance = np.linalg.norm(target_pos - current_pos)
                            print(f"[轨迹] 进度: {progress:.1f}%, 当前误差: {distance:.4f}m")
                    else:
                        # 轨迹执行完成
                        trajectory_active = False
                        elapsed = time.time() - trajectory_start_time
                        print(f"[POSE-IK] 轨迹执行完成，耗时: {elapsed:.2f}秒")
                        
                        # 更新目标位置/姿态（使用最后一个轨迹点，确保继续收敛到目标）
                        if trajectory_positions and trajectory_quaternions:
                            final_pos = trajectory_positions[-1].copy()
                            final_quat = trajectory_quaternions[-1].copy()
                        else:
                            final_pos = data.xpos[chains[trajectory_side]["ee"]].copy()
                            final_quat = data.xquat[chains[trajectory_side]["ee"]].copy()
                        
                        if trajectory_side == "left":
                            L_tgt = final_pos
                            L_ori_tgt = final_quat
                        else:
                            R_tgt = final_pos
                            R_ori_tgt = final_quat
                        if automation_current and automation_current.get("type") == "move_pose":
                            print("[automation] move_pose completed")
                            automation_current = None

                # ===== 常规控制更新 =====
                if goto and not physics_paused:
                    # 只有在没有执行轨迹时才进行常规的IK控制
                    if not trajectory_active:
                        if mode in ('left','both'):
                            ik_step_dynamic(
                                controller,
                                'left',
                                L_tgt,
                                target_quat=L_ori_tgt if keep_orientation else None,
                                max_iters=1,
                            )
                        if mode in ('right','both'):
                            ik_step_dynamic(
                                controller,
                                'right',
                                R_tgt,
                                target_quat=R_ori_tgt if keep_orientation else None,
                                max_iters=1,
                            )
                
                # 物理步进
                if not physics_paused:
                    controller.update_control()  # 夹爪保持功能已集成其中
                    mujoco.mj_step(model, data)

                # 录制和预览
                recorder.step(chains, L_tgt, R_tgt, controller)
                try:
                    import time as _t
                    preview.step(_t.perf_counter())
                except Exception:
                    pass

                viewer.sync()
                time.sleep(0.001)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
