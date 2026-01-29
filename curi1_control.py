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

from curi1_control_pkg.controllers import (
    TargetController,
    configure as controllers_configure,
    debug_gripper_contacts,
    gripper_troubleshooting,
    ik_step_dynamic,
    print_gripper_state,
    set_gripper,
    set_gripper_force_control,
    set_joint_delta,
)
from curi1_control_pkg.recording import CamPreview, Recorder
from curi1_control_pkg.automation import AutomationManager

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

controllers_configure(model, data, joint_to_actuator, _GRIP_ACT)

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
    automation_root = Path(__file__).resolve().parent / "automation"
    automation_root.mkdir(parents=True, exist_ok=True)
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
        (automation_root / "box_pose.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
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

    preview = CamPreview(
        model,
        data,
        cams=("top", "left_wrist", "right_wrist"),
        size=(320, 240),
        fps=15,
        suppress_qt_warnings=_suppress_qt_warnings,
    )
    record_root = "./datasets/sim_transfer_cube_scripted"
    recorder = Recorder(
        model,
        data,
        cams=("top", "left_wrist", "right_wrist"),
        fps=60,
        size=(640, 480),
        out_root=record_root,
        qpos_extractor=extract_14dim_qpos,
    )

    def get_targets():
        return L_tgt.copy(), R_tgt.copy()

    def update_targets(side, pos=None, quat=None):
        nonlocal L_tgt, R_tgt, L_ori_tgt, R_ori_tgt
        if side == "left":
            if pos is not None:
                L_tgt = pos.copy()
            if quat is not None:
                L_ori_tgt = quat.copy()
        else:
            if pos is not None:
                R_tgt = pos.copy()
            if quat is not None:
                R_ori_tgt = quat.copy()

    automation = AutomationManager(
        model=model,
        data=data,
        chains=chains,
        controller=controller,
        recorder=recorder,
        set_gripper_fn=lambda side, delta: set_gripper(side, delta, controller),
        set_gripper_force_fn=lambda side, delta: set_gripper_force_control(side, delta, controller),
        get_targets_fn=get_targets,
        set_target_fn=update_targets,
        automation_root=automation_root,
        trajectory_time_scale=TRAJECTORY_TIME_SCALE,
    )

    start_pose_trajectory = automation.start_pose_trajectory

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
                        if automation.is_trajectory_active:
                            automation.stop_trajectory()
                    
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
                automation.process_commands()

                # ===== 轨迹执行逻辑 =====
                automation.step_trajectory(physics_paused)

                # ===== 常规控制更新 =====
                if goto and not physics_paused:
                    # 只有在没有执行轨迹时才进行常规的IK控制
                    if not automation.is_trajectory_active:
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
