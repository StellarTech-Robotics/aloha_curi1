#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mujoco
import mujoco.viewer
import numpy as np
import time

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

MODEL_PATH = "bimanual_curi1_transfer_cube.xml"

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
        
    def set_target_from_ik(self, side, target_pos):
        """使用IK计算目标关节角度"""
        c = self.chains[side]
        
        # 多次迭代IK以获得更准确的解
        temp_qpos = data.qpos.copy()
        for _ in range(20):  # 增加迭代次数
            cur = data.xpos[c["ee"]].copy()
            e = target_pos - cur
            if np.linalg.norm(e) < 0.001:
                break
                
            J = self.numeric_jac(side)
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (1e-3**2)*np.eye(3), e)
            dq = np.clip(dq, -0.05, 0.05)
            
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

    def set_target_from_ik_pose(self, side, target_pos, target_quat, iters=40, damping=1e-4, pos_tol=1e-3, ori_tol=1e-2):
        """IK solve for both position and orientation. target_quat is (w,x,y,z)."""
        c = self.chains[side]
        temp_qpos = data.qpos.copy()

        for _ in range(iters):
            cur_p = data.xpos[c["ee"]].copy()
            cur_q = data.xquat[c["ee"]].copy()
            e_pos = target_pos - cur_p
            e_ori = quat_error_vec(target_quat, cur_q)
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_ori) < ori_tol:
                break

            J = self.numeric_jac_pose(side)
            JT = J.T
            H = JT @ J + (damping**2) * np.eye(J.shape[1])
            g = JT @ np.hstack([e_pos, e_ori])
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, np.hstack([e_pos, e_ori]), rcond=None)

            dq = np.clip(dq, -0.05, 0.05)
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
                self.integral_error[side][i] = np.clip(self.integral_error[side][i], -1.0, 1.0)
                
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
    
    def update_gripper_control(self):
        """更新夹爪控制（支持手动模式）"""
        gripper_actuators = _GRIP_ACT
        
        for side in ["left", "right"]:
            act_ids = gripper_actuators.get(side, [])
            if len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids):
                
                # ===== 关键修改：检查是否为手动模式 =====
                if self.gripper_manual_mode[side] and self.gripper_hold_positions[side] is not None:
                    # 手动模式：使用保持位置
                    target_pos = self.gripper_hold_positions[side]
                else:
                    # 自动模式：使用控制器目标
                    target_pos = self.gripper_targets[side]
                
                for aid in act_ids:
                    lo, hi = model.actuator_ctrlrange[aid]
                    data.ctrl[aid] = max(lo, min(hi, target_pos))
    
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

# ===== 改进的IK步进函数 =====
def ik_step_dynamic(controller, side, target, max_iters=5):
    """使用动力学兼容的方式进行IK控制"""
    controller.set_target_from_ik(side, target)
    
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

def set_joint_delta(joint_name, delta):
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
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.kind = None
        self._w = None
        try:
            import cv2  # type: ignore
            fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
            self._w = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
            self.kind = "cv2"
        except Exception:
            try:
                import imageio.v3 as iio  # type: ignore
                self._iio = iio
                self._w = iio.imopen(path, "w", plugin="ffmpeg", fps=self.fps)
                self.kind = "iio"
            except Exception as e:
                print("[record] ERROR: no video backend (cv2 or imageio).", e)
                self.kind = None

    def write(self, frame_rgb):
        if self._w is None:
            return
        if self.kind == "cv2":
            import cv2  # type: ignore
            bgr = frame_rgb[..., ::-1].copy()
            self._w.write(bgr)
        elif self.kind == "iio":
            self._w.write_frame(frame_rgb)

    def close(self):
        if self._w is None:
            return
        if self.kind == "cv2":
            self._w.release()
        elif self.kind == "iio":
            self._w.close()
        self._w = None

class Recorder:
    def __init__(self, model, data, cams=("top","left_wrist","right_wrist"), fps=30, size=(640,480), out_root="datasets"):
        self.model = model
        self.data = data
        self.cams = list(cams)
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        self.out_root = out_root
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

    def _ensure_renderers(self):
        maxW = int(getattr(self.model.vis.global_, "offwidth", 640))
        maxH = int(getattr(self.model.vis.global_, "offheight", 480))
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
        
        # CSV录制（可选，用于调试）
        self.csv_path = str(outdir / "states.csv")
        self.csv = open(self.csv_path, "w", newline="")
        import csv as _csv
        self.writer = _csv.writer(self.csv)
        nq = self.model.nq
        header = (["t", "frame"] +
                  [f"qpos_{i}" for i in range(nq)] +
                  ["Lx","Ly","Lz","Lqw","Lqx","Lqy","Lqz",
                   "Rx","Ry","Rz","Rqw","Rqx","Rqy","Rqz"] +
                  ["L_tgt_x","L_tgt_y","L_tgt_z","R_tgt_x","R_tgt_y","R_tgt_z"])
        self.writer.writerow(header)
        
        # 元数据
        self.meta = {
            "fps": self.fps,
            "size": self.size,
            "cameras": self.cams,
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
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
                    
                    # 保存qpos和qvel
                    if len(self.qpos_buffer) > 0:
                        qpos_data = np.array(self.qpos_buffer, dtype=np.float32)
                        obs_grp.create_dataset('qpos', data=qpos_data,
                                              compression='gzip', compression_opts=4)
                        print(f"  - qpos: {qpos_data.shape}")
                    
                    if len(self.qvel_buffer) > 0:
                        qvel_data = np.array(self.qvel_buffer, dtype=np.float32)
                        obs_grp.create_dataset('qvel', data=qvel_data,
                                              compression='gzip', compression_opts=4)
                        print(f"  - qvel: {qvel_data.shape}")
                    
                    # 保存action（对于仿真，action通常就是目标qpos）
                    if len(self.action_buffer) > 0:
                        action_data = np.array(self.action_buffer, dtype=np.float32)
                        h5.create_dataset('action', data=action_data,
                                        compression='gzip', compression_opts=4)
                        print(f"  - action: {action_data.shape}")
                    
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
        self.enabled = False

    def step(self, chains, L_tgt, R_tgt):
        if not self.enabled or self.t0 is None:
            return
        t_rel = time.time() - self.t0
        if t_rel - self.last + 1e-9 < self.interval:
            return
        self.last += self.interval
        
        # 渲染并缓存图像
        for cam, rend in self.renderers.items():
            try:
                rend.update_scene(self.data, camera=cam)
                rgb = rend.render()
                # 保存到视频
                if cam in self.videos:
                    self.videos[cam].write(rgb)
                # 缓存到内存（用于HDF5）
                self.image_buffers[cam].append(rgb.copy())
            except Exception as e:
                print(f"[record] camera {cam}: {e}")
        
        # 缓存状态数据
        self.qpos_buffer.append(self.data.qpos[:].copy())
        self.qvel_buffer.append(self.data.qvel[:].copy())
        
        # 对于action，在仿真中通常是下一时刻的目标qpos
        # 这里我们使用控制器的目标关节角度
        # 需要从controller获取完整的目标qpos
        action = self.data.qpos[:].copy()  # 简化处理：使用当前qpos作为action
        self.action_buffer.append(action)
        
        # CSV记录（用于调试）
        if self.csv and self.writer:
            Lp = self.data.xpos[chains["left"]["ee"]].copy()
            Lq = self.data.xquat[chains["left"]["ee"]].copy()
            Rp = self.data.xpos[chains["right"]["ee"]].copy()
            Rq = self.data.xquat[chains["right"]["ee"]].copy()
            row = [t_rel, self.frame_id] + [float(x) for x in self.data.qpos[:]] + \
                  [*Lp.tolist(), *Lq.tolist(), *Rp.tolist(), *Rq.tolist(),
                   float(L_tgt[0]), float(L_tgt[1]), float(L_tgt[2]),
                   float(R_tgt[0]), float(R_tgt[1]), float(R_tgt[2])]
            try:
                self.writer.writerow(row)
            except Exception:
                pass
        
        self.frame_id += 1

# ================== Camera preview ==================
class CamPreview:
    def __init__(self, model, data, cams=("top","left_wrist","right_wrist"), size=(320,240), fps=15, window="cams"):
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
        try:
            import cv2 as _cv2
            self._cv2 = _cv2
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
            print("[preview] cv2 not available; cannot show preview.")
            return
        self.enabled = not self.enabled
        if not self.enabled:
            try:
                self._cv2.destroyWindow(self.window)
            except Exception:
                pass
        print("[preview] enabled =", self.enabled)

    def step(self, t_now):
        if not self.enabled or self._cv2 is None:
            return
        if t_now - self.last_t + 1e-9 < self.interval:
            return
        self.last_t += self.interval

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
        self._cv2.imshow(self.window, panel)
        self._cv2.waitKey(1)

HELP = """
[Viewer+Terminal Controls] (keep TERMINAL focused)
  h    : help
  q    : quit
  0    : reset to keyframe 0
  [/]  : PID gains down/up
  -/=  : pos step down/up
  1/2/3: LEFT / RIGHT / BOTH
  g    : toggle go-to IK mode
  w/s, a/d, r/f : +Y/-Y, -X/+X, +Z/-Z (world)
  i/k, j/l, u/h : right arm +Y/-Y, -X/+X, +Z/-Z (world)
  
  === 夹爪控制 ===
  z/x  : 左夹爪闭合/张开 (普通模式)
  n/m  : 右夹爪闭合/张开 (普通模式)
  Z/X  : 左夹爪闭合/张开 (力控制模式，会检测碰撞)
  N/M  : 右夹爪闭合/张开 (力控制模式，会检测碰撞)
  
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
- 松开按键后自动保持位置，不会回弹
- 按0重置时会释放夹爪手动控制
- 使用c键和v键来调试碰撞和距离问题
"""

def main():
    chains = find_arm_chain()
    print_chain(chains)

    # 创建控制器
    controller = TargetController(chains, kp=100.0, kd=20.0, ki=0.5)
    
    mujoco.mj_forward(model, data)
    L_tgt = data.xpos[chains["left"]["ee"]].copy()
    R_tgt = data.xpos[chains["right"]["ee"]].copy()

    mode = "left"
    step = 0.001
    goto = True
    physics_paused = False
    
    # ===== 新增：轨迹执行相关变量 =====
    trajectory_active = False
    trajectory_side = None
    trajectory_positions = []
    trajectory_quaternions = []
    trajectory_index = 0
    trajectory_start_time = None
    trajectory_duration = 0.0
    
    preview = CamPreview(model, data, cams=("top","left_wrist","right_wrist"), size=(320,240), fps=15)
    recorder = Recorder(model, data, cams=("top","left_wrist","right_wrist"), fps=30, size=(640,480))

    # ===== 新增：轨迹生成函数 =====
    def generate_trajectory(side, target_pos, target_quat, seconds=2.0, fps=60):
        """生成平滑的轨迹点"""
        steps = max(1, int(seconds * fps))
        c = controller.chains[side]
        p0 = data.xpos[c["ee"]].copy()
        q0 = data.xquat[c["ee"]].copy()
        
        positions = []
        quaternions = []
        
        for i in range(steps):
            alpha = (i+1) / steps
            # 使用5次多项式插值，实现更平滑的加减速
            # alpha_smooth = 6 * alpha**5 - 15 * alpha**4 + 10 * alpha**3
            # 或者使用简单的正弦曲线插值
            alpha_smooth = 0.5 * (1 - np.cos(np.pi * alpha))
            
            p = (1.0-alpha_smooth) * p0 + alpha_smooth * target_pos
            q = quat_slerp(q0, target_quat, alpha_smooth)
            
            positions.append(p.copy())
            quaternions.append(q.copy())
        
        return positions, quaternions

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
                    if ch in ('b','B'):
                        print(HELP)
                    elif ch == 'q':
                        return
                    elif ch == '0':
                        mujoco.mj_resetDataKeyframe(model, data, 0)
                        data.qvel[:] = 0; data.act[:] = 0
                        mujoco.mj_forward(model, data)
                        L_tgt = data.xpos[chains["left"]["ee"]].copy()
                        R_tgt = data.xpos[chains["right"]["ee"]].copy()
                        # 重置时释放夹爪手动控制
                        controller.set_gripper_manual_mode("left", False)
                        controller.set_gripper_manual_mode("right", False)
                        print("[reset] keyframe 0, gripper manual control released")
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
                    elif ch == 'w':
                        if mode in ('left','both'): L_tgt[1] += step
                    elif ch == 's':
                        if mode in ('left','both'): L_tgt[1] -= step
                    elif ch == 'a':
                        if mode in ('left','both'): L_tgt[0] -= step
                    elif ch == 'd':
                        if mode in ('left','both'): L_tgt[0] += step
                    elif ch == 'r':
                        if mode in ('left','both'): L_tgt[2] += step
                    elif ch == 'f':
                        if mode in ('left','both'): L_tgt[2] -= step
                    elif ch == 'i':
                        if mode in ('right','both'): R_tgt[1] += step
                    elif ch == 'k':
                        if mode in ('right','both'): R_tgt[1] -= step
                    elif ch == 'j':
                        if mode in ('right','both'): R_tgt[0] -= step
                    elif ch == 'l':
                        if mode in ('right','both'): R_tgt[0] += step
                    elif ch == 'u':
                        if mode in ('right','both'): R_tgt[2] += step
                    elif ch == 'h':
                        if mode in ('right','both'): R_tgt[2] -= step
                    
                    # ===== 夹爪控制（普通模式）=====
                    elif ch == 'z': 
                        set_gripper("left", -0.005, controller)
                        print_gripper_state()
                        print("[gripper] Left manual mode ON (normal)")
                    elif ch == 'x': 
                        set_gripper("left", +0.005, controller)
                        print_gripper_state()
                        print("[gripper] Left manual mode ON (normal)")
                    elif ch == 'n': 
                        set_gripper("right", -0.005, controller)
                        print_gripper_state()
                        print("[gripper] Right manual mode ON (normal)")
                    elif ch == 'm': 
                        set_gripper("right", +0.005, controller)
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
                            line = input("格式: side x y z roll pitch yaw(deg) duration(s)\n例如: right 0.20 -0.60 0.65 0 0 0 2\n> ")
                            parts = line.strip().split()
                            if len(parts) != 8:
                                print("输入格式错误，需要8个参数")
                                continue
                                
                            side, x, y, z, r, p_, yw, dur = parts
                            pos  = np.array([float(x), float(y), float(z)], dtype=np.float64)
                            quat = rpy_to_quat(np.deg2rad(float(r)), np.deg2rad(float(p_)), np.deg2rad(float(yw)))
                            
                            # 生成轨迹
                            trajectory_positions, trajectory_quaternions = generate_trajectory(
                                side.lower(), pos, quat, seconds=float(dur), fps=60
                            )
                            
                            # 激活轨迹执行
                            trajectory_active = True
                            trajectory_side = side.lower()
                            trajectory_index = 0
                            trajectory_start_time = time.time()
                            trajectory_duration = float(dur)
                            
                            print(f"[POSE-IK] 开始执行轨迹: {side} -> pos={pos}, rpy=({r},{p_},{yw})deg, 持续{dur}秒")
                            
                        except Exception as e:
                            print("输入解析失败:", e)

                    elif ch == 'Q':  # 输入位置 + 四元数
                        try:
                            line = input("格式: side x y z qw qx qy qz duration(s)\n例如: left 0.15 -0.55 0.70 1 0 0 0 1.5\n> ")
                            parts = line.strip().split()
                            if len(parts) != 9:
                                print("输入格式错误，需要9个参数")
                                continue
                                
                            side, x, y, z, qw, qx, qy, qz, dur = parts
                            pos  = np.array([float(x), float(y), float(z)], dtype=np.float64)
                            quat = quat_normalize(np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float64))
                            
                            # 生成轨迹
                            trajectory_positions, trajectory_quaternions = generate_trajectory(
                                side.lower(), pos, quat, seconds=float(dur), fps=60
                            )
                            
                            # 激活轨迹执行
                            trajectory_active = True
                            trajectory_side = side.lower()
                            trajectory_index = 0
                            trajectory_start_time = time.time()
                            trajectory_duration = float(dur)
                            
                            print(f"[POSE-IK] 开始执行轨迹: {side} -> pos={pos}, quat={quat}, 持续{dur}秒")
                            
                        except Exception as e:
                            print("输入解析失败:", e)

                    elif ch == 'E':  # Emergency stop - 紧急停止当前轨迹
                        if trajectory_active:
                            trajectory_active = False
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
                        
                        # 更新目标位置（用于后续的手动控制）
                        if trajectory_side == "left":
                            L_tgt = data.xpos[chains["left"]["ee"]].copy()
                        else:
                            R_tgt = data.xpos[chains["right"]["ee"]].copy()

                # ===== 常规控制更新 =====
                if goto and not physics_paused:
                    # 只有在没有执行轨迹时才进行常规的IK控制
                    if not trajectory_active:
                        if mode in ('left','both'):
                            ik_step_dynamic(controller, 'left', L_tgt, max_iters=1)
                        if mode in ('right','both'):
                            ik_step_dynamic(controller, 'right', R_tgt, max_iters=1)
                
                # 物理步进
                if not physics_paused:
                    controller.update_control()  # 夹爪保持功能已集成其中
                    mujoco.mj_step(model, data)
                
                # 录制和预览
                recorder.step(chains, L_tgt, R_tgt)
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