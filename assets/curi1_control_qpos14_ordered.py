#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mujoco
import mujoco.viewer
import numpy as np
import time

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

def print_joint_debug(joint_name):
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    except Exception:
        jid = -0.05 # -1
    if jid > 0 or jid < -1:
        print(f"[debug] joint '{joint_name}' not found"); return
    jtype = int(model.jnt_type[jid]); jtype_s = {0:'free',1:'ball',2:'slide',3:'hinge'}.get(jtype,str(jtype))
    adr = model.jnt_qposadr[jid]; dadr = model.jnt_dofadr[jid]
    lo,hi = model.jnt_range[jid]; axis = model.jnt_axis[jid]
    qpos = float(data.qpos[adr]); qvel=float(data.qvel[dadr])
    # find mapped actuators
    acts=[]; trn=getattr(model,'actuator_trnid',None)
    if trn is not None:
        for aid in range(model.nu):
            if trn[aid][0]==jid:
                aname=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,aid)
                cr_lo,cr_hi=model.actuator_ctrlrange[aid]
                ctrl=float(data.ctrl[aid]); gear=model.actuator_gear[aid].copy()
                acts.append((aid,aname,ctrl,(cr_lo,cr_hi),gear))
    print("\\n[joint-debug] ----")
    print(f"name={joint_name} id={jid} type={jtype_s} axis={np.round(axis,4)}")
    print(f"qpos={qpos:.6f} qvel={qvel:.6f} jnt_range=[{lo:.6f},{hi:.6f}]")
    if acts:
        for (aid,aname,ctrl,(cr_lo,cr_hi),gear) in acts:
            print(f"  actuator id={aid} name={aname}  ctrl={ctrl:.6f}  ctrlrange=[{cr_lo:.6f},{cr_hi:.6f}]  gear={np.round(gear,4)}")
    else:
        print("  (no actuator mapped to this joint)")
    print("--------------\\n")

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
        import time, json, os
        from pathlib import Path as _Path
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = _Path(self.out_root) / f"run_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)
        self._ensure_renderers()
        # === Build qpos/qvel indices to keep (both arms + 1 gripper scalar per side = 14 dims) ===
        try:
            import mujoco
            # 1) arm joints: take first 6 from each side (as find_arm_chain already sorted/limited)
            left_qadr  = [int(x) for x in chains['left']['qadr'][:6]]
            right_qadr = [int(x) for x in chains['right']['qadr'][:6]]
            left_jids  = [int(x) for x in chains['left']['jids'][:6]]
            right_jids = [int(x) for x in chains['right']['jids'][:6]]

            # 2) pick one representative gripper joint each side (first available in the config list)
            def pick_gripper_joint(name_list):
                for nm in name_list:
                    try:
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                        if jid >= 0:
                            return int(jid), nm
                    except Exception:
                        pass
                return None, None

            # These lists should be defined in the file's global scope
            try:
                from __main__ import LEFT_GRIPPER_JOINTS, RIGHT_GRIPPER_JOINTS
            except Exception:
                # Fallback: hard-coded common names
                LEFT_GRIPPER_JOINTS  = ['Joint_finger1', 'Joint_finger2']
                RIGHT_GRIPPER_JOINTS = ['r_Joint_finger1', 'r_Joint_finger2']

            l_g_jid, l_g_name = pick_gripper_joint(LEFT_GRIPPER_JOINTS)
            r_g_jid, r_g_name = pick_gripper_joint(RIGHT_GRIPPER_JOINTS)

            add_qadr = []
            add_names = []
            add_vadr = []
            if l_g_jid is not None:
                add_qadr.append(int(self.model.jnt_qposadr[l_g_jid]))
                add_vadr.append(int(self.model.jnt_dofadr[l_g_jid]))
                add_names.append(l_g_name or 'left_gripper')
            if r_g_jid is not None:
                add_qadr.append(int(self.model.jnt_qposadr[r_g_jid]))
                add_vadr.append(int(self.model.jnt_dofadr[r_g_jid]))
                add_names.append(r_g_name or 'right_gripper')

            # Final indices & names in order: left6, right6, l_grip, r_grip
            self.qpos_index = left_qadr + ([int(self.model.jnt_qposadr[l_g_jid])] if l_g_jid is not None else []) + right_qadr + ([int(self.model.jnt_qposadr[r_g_jid])] if r_g_jid is not None else [])
            self.qvel_index = [int(self.model.jnt_dofadr[j]) for j in left_jids] + ([int(self.model.jnt_dofadr[l_g_jid])] if l_g_jid is not None else []) + [int(self.model.jnt_dofadr[j]) for j in right_jids] + ([int(self.model.jnt_dofadr[r_g_jid])] if r_g_jid is not None else [])
            # Names aligned
            self.saved_joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in left_jids] + (([l_g_name or 'left_gripper']) if l_g_jid is not None else []) + [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in right_jids] + (([r_g_name or 'right_gripper']) if r_g_jid is not None else [])

        except Exception as _e:
            # Fallback: keep all (avoid crash)
            self.qpos_index = None
            self.qvel_index = None
            self.saved_joint_names = None

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
                import numpy as np
                
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
                    if self.saved_joint_names is not None:
                        import numpy as _np
                        import h5py as _h5
                        _dt = _h5.string_dtype(encoding='utf-8')
                        h5.attrs.create('joint_names', _np.array(self.saved_joint_names, dtype=_dt))
                    if self.qpos_index is not None:
                        import numpy as _np
                        h5.attrs.create('qpos_index', _np.array(self.qpos_index, dtype=_np.int32))

                    
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
        import time, numpy as np
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
        self.qpos_buffer.append((self.data.qpos[self.qpos_index] if self.qpos_index is not None else self.data.qpos[:]).copy())
        self.qvel_buffer.append((self.data.qvel[self.qvel_index] if self.qvel_index is not None else self.data.qvel[:]).copy())
        
        # 对于action，在仿真中通常是下一时刻的目标qpos
        # 这里我们使用控制器的目标关节角度
        # 需要从controller获取完整的目标qpos
        action = (self.data.qpos[self.qpos_index] if self.qpos_index is not None else self.data.qpos[:]).copy()  # 使用筛选后的维度作为action
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
    
    preview = CamPreview(model, data, cams=("top","left_wrist","right_wrist"), size=(320,240), fps=15)
    recorder = Recorder(model, data, cams=("top","left_wrist","right_wrist"), fps=30, size=(640,480))

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
        print("[INFO] 碰撞检测已可用，物理模拟正常运行")
        print("[INFO] 夹爪保持功能已完全集成到控制器中")
        print("[INFO] 增加了碰撞检测和力控制功能")
        
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
                        model.opt.timestep *= 0.5
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
                    
                    # Head & platform joints
                    elif ch == ',': set_joint_delta('head_joint1', -0.02)
                    elif ch == '.': set_joint_delta('head_joint1', +0.02)
                    elif ch == ';': set_joint_delta('head_joint2', -0.02)
                    elif ch == ':': set_joint_delta('head_joint2', +0.02)
                    elif ch == '$': set_joint_delta('l_joint4', -0.02)
                    elif ch == '%': set_joint_delta('l_joint4', +0.02)
                    elif ch == '4': set_joint_delta('l_joint5', -0.02)
                    elif ch == '5': set_joint_delta('l_joint5', +0.02)
                    elif ch == '6': set_joint_delta('l_joint6', -0.02)
                    elif ch == '7': set_joint_delta('l_joint6', +0.02)
                    elif ch == '(': set_joint_delta('r_joint4', -0.02)
                    elif ch == ')': set_joint_delta('r_joint4', +0.02)
                    elif ch == '8': set_joint_delta('r_joint5', -0.02)
                    elif ch == '9': set_joint_delta('r_joint5', +0.02)
                    elif ch == '¥': set_joint_delta('r_joint6', -0.02)
                    elif ch == '^': set_joint_delta('r_joint6', +0.02)
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
                    elif ch == 'p': print_joint_debug('platform_joint')
                    elif ch == 'P':
                        physics_paused = not physics_paused
                        print(f"[physics] {'PAUSED' if physics_paused else 'RUNNING'}")

                # 更新控制
                if goto and not physics_paused:
                    if mode in ('left','both'):
                        ik_step_dynamic(controller, 'left', L_tgt, max_iters=1)
                    if mode in ('right','both'):
                        ik_step_dynamic(controller, 'right', R_tgt, max_iters=1)
                
                # 物理步进
                if not physics_paused:
                    # ===== 关键：控制器现在内部处理夹爪手动模式 =====
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