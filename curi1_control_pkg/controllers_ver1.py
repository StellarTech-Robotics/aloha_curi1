"""Controllers and gripper helpers for curi1_control."""
from __future__ import annotations

import numpy as np
import mujoco

from .pose_utils import quat_error_vec, quat_normalize

_MODEL = None
_DATA = None
_JOINT_TO_ACT = {}
_GRIP_ACT = {}

FINGER_JOINT_NAMES = {
    "left": ["Joint_finger1", "Joint_finger2"],
    "right": ["r_Joint_finger1", "r_Joint_finger2"],
}


def configure(model, data, joint_to_actuator, gripper_actuators):
    """Share MuJoCo buffers and mappings with this module."""
    global _MODEL, _DATA, _JOINT_TO_ACT, _GRIP_ACT
    _MODEL = model
    _DATA = data
    _JOINT_TO_ACT = joint_to_actuator or {}
    _GRIP_ACT = gripper_actuators or {}


def _mj_forward():
    mujoco.mj_forward(_MODEL, _DATA)


def debug_gripper_contacts(side="left"):
    """检查夹爪是否与物体发生碰撞"""
    finger_names = FINGER_JOINT_NAMES["left"] if side == "left" else FINGER_JOINT_NAMES["right"]

    print(f"\n=== {side.upper()} 夹爪碰撞检测 ===")
    for finger_name in finger_names:
        try:
            finger_jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, finger_name)
            if finger_jid >= 0:
                finger_body_id = _MODEL.jnt_bodyid[finger_jid]
                finger_body_name = mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_BODY, finger_body_id)
                print(f"手指 {finger_name} -> body: {finger_body_name}")
        except Exception:
            continue

    contact_count = 0
    for i in range(_DATA.ncon):
        contact = _DATA.contact[i]
        geom1_name = mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        geom2_name = mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        is_contact = any(
            fn in geom1_name or fn in geom2_name
            for fn in finger_names
        )
        if is_contact:
            contact_count += 1
            print(f"  碰撞: {geom1_name} <-> {geom2_name}")

    print(f"总碰撞数: {contact_count}\n")
    return contact_count


class TargetController:
    """用于平滑跟踪目标位置的控制器，支持夹爪手动控制"""

    def __init__(self, chains, kp=50.0, kd=10.0, ki=0.5):
        self.chains = chains
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.joint_to_actuator = _JOINT_TO_ACT

        self.target_qpos = {}
        for side in ["left", "right"]:
            self.target_qpos[side] = np.zeros(len(chains[side]["qadr"]))
            for i, qadr in enumerate(chains[side]["qadr"]):
                self.target_qpos[side][i] = _DATA.qpos[qadr]

        self.integral_error = {"left": np.zeros(6), "right": np.zeros(6)}
        self.gripper_targets = {"left": 0.0, "right": 0.0}

        self.gripper_manual_mode = {"left": False, "right": False}
        self.gripper_hold_positions = {"left": None, "right": None}

        self.gripper_servo_lock = {"left": False, "right": False}
        self.gripper_lock_position = {"left": None, "right": None}
        self.gripper_servo_kp = 200.0

        self.gripper_force_lock = {"left": False, "right": False}
        self.gripper_force_target = {"left": 0.0, "right": 0.0}
        self.gripper_force_dofs = {"left": [], "right": []}
        for side in ("left", "right"):
            dofs = []
            for joint_name in FINGER_JOINT_NAMES[side]:
                jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid >= 0:
                    dofs.append(_MODEL.jnt_dofadr[jid])
                else:
                    dofs.append(-1)
            self.gripper_force_dofs[side] = dofs

        self.ee_pose_targets = {
            "left": {"pos": None, "quat": None, "kp_pos": 4.0, "kp_ori": 2.5},
            "right": {"pos": None, "quat": None, "kp_pos": 4.0, "kp_ori": 2.5},
        }

    def _finger_joint_names(self, side):
        return FINGER_JOINT_NAMES["left"] if side == "left" else FINGER_JOINT_NAMES["right"]

    def set_target_from_ik(self, side, target_pos, pos_tol=1e-4, max_iters=80):
        c = self.chains[side]

        temp_qpos = _DATA.qpos.copy()
        for _ in range(max_iters):
            cur = _DATA.xpos[c["ee"]].copy()
            e = target_pos - cur
            err = np.linalg.norm(e)
            if err < pos_tol:
                break

            J = self.numeric_jac(side)
            JJt = J @ J.T
            reg = max(5e-6, min(3e-4, err * 0.4))
            dq = J.T @ np.linalg.solve(JJt + (reg**2) * np.eye(3), e)
            clip = 0.04 if err > 1e-2 else (0.03 if err > 3e-3 else 0.015)
            dq = np.clip(dq, -clip, clip)

            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                v = _DATA.qpos[qadr] + dq[i]
                rmin, rmax = _MODEL.jnt_range[jid]
                if rmin < rmax:
                    v = max(rmin, min(rmax, v))
                _DATA.qpos[qadr] = v
            _mj_forward()

        for i, qadr in enumerate(c["qadr"]):
            self.target_qpos[side][i] = _DATA.qpos[qadr]

        _DATA.qpos[:] = temp_qpos
        _mj_forward()
        self.set_ee_pose_target(side, target_pos)

    def numeric_jac_pose(self, side, eps=1e-5):
        c = self.chains[side]
        n = len(c["qadr"])
        Jp = np.zeros((3, n), dtype=np.float64)
        Jo = np.zeros((3, n), dtype=np.float64)
        qbackup = _DATA.qpos.copy()
        _mj_forward()

        p0 = _DATA.xpos[c["ee"]].copy()
        q0 = _DATA.xquat[c["ee"]].copy()

        for k, adr in enumerate(c["qadr"]):
            _DATA.qpos[:] = qbackup
            _DATA.qpos[adr] += eps
            _mj_forward()
            p_plus = _DATA.xpos[c["ee"]].copy()
            q_plus = _DATA.xquat[c["ee"]].copy()

            _DATA.qpos[:] = qbackup
            _DATA.qpos[adr] -= eps
            _mj_forward()
            p_minus = _DATA.xpos[c["ee"]].copy()
            q_minus = _DATA.xquat[c["ee"]].copy()

            Jp[:, k] = (p_plus - p_minus) / (2.0 * eps)
            e_plus = quat_error_vec(q_plus, q0)
            e_minus = quat_error_vec(q_minus, q0)
            Jo[:, k] = (e_plus - e_minus) / (2.0 * eps)

        _DATA.qpos[:] = qbackup
        _mj_forward()
        return np.vstack([Jp, Jo])

    def set_target_from_ik_pose(self, side, target_pos, target_quat, iters=90, damping=5e-5, pos_tol=1.5e-4, ori_tol=3e-3):
        c = self.chains[side]
        temp_qpos = _DATA.qpos.copy()

        for _ in range(iters):
            cur_p = _DATA.xpos[c["ee"]].copy()
            cur_q = _DATA.xquat[c["ee"]].copy()
            e_pos = target_pos - cur_p
            e_ori = quat_error_vec(target_quat, cur_q)
            err_pos = np.linalg.norm(e_pos)
            err_ori = np.linalg.norm(e_ori)
            if err_pos < pos_tol and err_ori < ori_tol:
                break

            J = self.numeric_jac_pose(side)
            JT = J.T
            reg = max(5e-6, min(3e-4, err_pos * 0.35))
            H = JT @ J + (max(reg, damping) ** 2) * np.eye(J.shape[1])
            g = JT @ np.hstack([e_pos, e_ori])
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, np.hstack([e_pos, e_ori]), rcond=None)

            clip = 0.035 if err_pos > 3e-3 else 0.015
            dq = np.clip(dq, -clip, clip)
            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                v = float(_DATA.qpos[qadr] + dq[i])
                rmin, rmax = _MODEL.jnt_range[jid]
                if rmin < rmax:
                    v = max(rmin, min(rmax, v))
                _DATA.qpos[qadr] = v
            _mj_forward()

        for i, qadr in enumerate(c["qadr"]):
            self.target_qpos[side][i] = _DATA.qpos[qadr]

        _DATA.qpos[:] = temp_qpos
        _mj_forward()
        self.set_ee_pose_target(side, target_pos, target_quat)

    def numeric_jac(self, side, eps=1e-5):
        c = self.chains[side]
        J = np.zeros((3, len(c["qadr"])))
        qbackup = _DATA.qpos.copy()
        _mj_forward()

        for k, adr in enumerate(c["qadr"]):
            _DATA.qpos[:] = qbackup
            _DATA.qpos[adr] += eps
            _mj_forward()
            p_plus = _DATA.xpos[c["ee"]].copy()

            _DATA.qpos[:] = qbackup
            _DATA.qpos[adr] -= eps
            _mj_forward()
            p_minus = _DATA.xpos[c["ee"]].copy()

            J[:, k] = (p_plus - p_minus) / (2 * eps)

        _DATA.qpos[:] = qbackup
        _mj_forward()
        return J

    def set_gripper_manual_mode(self, side, enabled, hold_position=None):
        self.gripper_manual_mode[side] = enabled
        if enabled and hold_position is not None:
            self.gripper_hold_positions[side] = hold_position
        elif not enabled:
            self.gripper_hold_positions[side] = None
            self.gripper_force_lock[side] = False
            self.gripper_force_target[side] = 0.0

    def set_ee_pose_target(self, side, pos=None, quat=None, kp_pos=None, kp_ori=None):
        target = self.ee_pose_targets[side]
        if pos is not None:
            target["pos"] = np.asarray(pos, dtype=np.float64).copy()
        if quat is not None:
            target["quat"] = quat_normalize(np.asarray(quat, dtype=np.float64))
        if kp_pos is not None:
            target["kp_pos"] = float(kp_pos)
        if kp_ori is not None:
            target["kp_ori"] = float(kp_ori)

    def clear_ee_pose_target(self, side):
        self.ee_pose_targets[side]["pos"] = None
        self.ee_pose_targets[side]["quat"] = None

    def update_gripper_hold_position(self, side):
        if not self.gripper_manual_mode[side]:
            return

        for jn in self._finger_joint_names(side):
            jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                adr = _MODEL.jnt_qposadr[jid]
                self.gripper_hold_positions[side] = float(_DATA.qpos[adr])
                break

    def toggle_gripper_servo_lock(self, side):
        self.gripper_servo_lock[side] = not self.gripper_servo_lock[side]

        if self.gripper_servo_lock[side]:
            jnames = self._finger_joint_names(side)
            jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, jnames[0])
            if jid >= 0:
                adr = _MODEL.jnt_qposadr[jid]
                current_pos = _DATA.qpos[adr]
                self.gripper_lock_position[side] = current_pos
                print(f"[SERVO LOCK {side.upper()}] 已锁定位置: {current_pos:.4f}m")
                print(f"[SERVO LOCK {side.upper()}] 使用 kp={self.gripper_servo_kp} 的位置伺服控制")
                print(f"[提示] {side}夹爪已锁定当前位置，将持续施加抓取力")
                print(f"[提示] 再次按 Ctrl+{'G' if side=='left' else 'H'} 解除锁定")
        else:
            self.gripper_lock_position[side] = None
            print(f"[SERVO LOCK {side.upper()}] 已解锁，恢复手动控制")
            print(f"[提示] {side}夹爪已解除锁定")

        return self.gripper_servo_lock[side]

    def toggle_gripper_force_lock(self, side, force_newton=10.0):
        self.gripper_force_lock[side] = not self.gripper_force_lock[side]

        if self.gripper_force_lock[side]:
            self.gripper_servo_lock[side] = False
            self.gripper_lock_position[side] = None
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
        self._apply_ee_pose_targets()
        for side in ["left", "right"]:
            c = self.chains[side]

            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                error = self.target_qpos[side][i] - _DATA.qpos[qadr]
                dadr = _MODEL.jnt_dofadr[jid]
                vel = _DATA.qvel[dadr] if dadr >= 0 else 0.0
                self.integral_error[side][i] += error * 0.001
                self.integral_error[side][i] = np.clip(self.integral_error[side][i], -0.5, 0.5)
                control = self.kp * error - self.kd * vel + self.ki * self.integral_error[side][i]

                if jid in self.joint_to_actuator:
                    aid = self.joint_to_actuator[jid]
                    clo, chi = _MODEL.actuator_ctrlrange[aid]
                    _DATA.ctrl[aid] = np.clip(self.target_qpos[side][i], clo, chi)

        self.update_gripper_control()
        self.apply_gripper_force_lock()

    def _apply_ee_pose_targets(self):
        for side in ("left", "right"):
            target = self.ee_pose_targets[side]
            target_pos = target.get("pos")
            target_quat = target.get("quat")
            if target_pos is None and target_quat is None:
                continue

            c = self.chains[side]
            cur_pos = _DATA.xpos[c["ee"]].copy()
            cur_quat = _DATA.xquat[c["ee"]].copy()
            err_vec = np.zeros(6, dtype=np.float64)
            has_error = False

            if target_pos is not None:
                e_pos = target_pos - cur_pos
                err_vec[:3] = target["kp_pos"] * e_pos
                has_error = has_error or (np.linalg.norm(e_pos) > 5e-5)
            if target_quat is not None:
                e_ori = quat_error_vec(target_quat, cur_quat)
                err_vec[3:] = target["kp_ori"] * e_ori
                has_error = has_error or (np.linalg.norm(e_ori) > 1e-4)

            if not has_error:
                continue

            J = self.numeric_jac_pose(side)
            JT = J.T
            lam = 1e-3
            H = JT @ J + (lam ** 2) * np.eye(J.shape[1])
            try:
                dq = np.linalg.solve(H, JT @ err_vec)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, err_vec, rcond=None)
            dq = np.clip(dq, -0.02, 0.02)

            for idx, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                val = self.target_qpos[side][idx] + dq[idx]
                rmin, rmax = _MODEL.jnt_range[jid]
                if rmin < rmax:
                    val = max(rmin, min(rmax, val))
                self.target_qpos[side][idx] = val

    def update_gripper_control(self):
        for side in ("left", "right"):
            act_ids = _GRIP_ACT.get(side, [])
            if not (len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids)):
                continue

            if self.gripper_servo_lock[side] and self.gripper_lock_position[side] is not None:
                target_pos = self.gripper_lock_position[side]
            elif self.gripper_manual_mode[side] and self.gripper_hold_positions[side] is not None:
                target_pos = self.gripper_hold_positions[side]
            else:
                target_pos = self.gripper_targets[side]

            for aid in act_ids:
                lo, hi = _MODEL.actuator_ctrlrange[aid]
                _DATA.ctrl[aid] = max(lo, min(hi, target_pos))

    def apply_gripper_force_lock(self):
        for side in ("left", "right"):
            dofs = self.gripper_force_dofs[side]
            if self.gripper_force_lock[side] and self.gripper_force_target[side] > 0.0:
                force = -self.gripper_force_target[side]
                for dof in dofs:
                    if dof is not None and dof >= 0:
                        _DATA.qfrc_applied[dof] = force
            else:
                for dof in dofs:
                    if dof is not None and dof >= 0:
                        _DATA.qfrc_applied[dof] = 0.0

    def set_gripper_target(self, side, value):
        if self.gripper_manual_mode[side]:
            return

        for jn in self._finger_joint_names(side):
            jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                lo, hi = _MODEL.jnt_range[jid]
                self.gripper_targets[side] = max(lo, min(hi, value))
                break

    def set_14dim_target_qpos(self, mobile_aloha_qpos):
        for i in range(6):
            self.target_qpos["left"][i] = mobile_aloha_qpos[i]

        self.gripper_targets["left"] = mobile_aloha_qpos[6]

        for i in range(6):
            self.target_qpos["right"][i] = mobile_aloha_qpos[7 + i]

        self.gripper_targets["right"] = mobile_aloha_qpos[13]


def ik_step_dynamic(controller, side, target_pos, target_quat=None, max_iters=5):
    """Run a short IK tracking burst under Mujoco dynamics."""
    if target_quat is not None:
        controller.set_target_from_ik_pose(side, target_pos, quat_normalize(target_quat))
    else:
        controller.set_target_from_ik(side, target_pos)

    for _ in range(max_iters):
        controller.update_control()
        mujoco.mj_step(_MODEL, _DATA)
    controller.set_ee_pose_target(side, target_pos, target_quat)


def set_gripper(side, delta, controller):
    controller.set_gripper_manual_mode(side, True)
    jnames = FINGER_JOINT_NAMES["left"] if side == "left" else FINGER_JOINT_NAMES["right"]
    targets = []
    for jn in jnames:
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        adr = _MODEL.jnt_qposadr[jid]
        lo, hi = _MODEL.jnt_range[jid]
        v = float(_DATA.qpos[adr]) + float(delta)
        if lo < hi:
            v = max(lo, min(hi, v))
        _DATA.qpos[adr] = v
        targets.append((adr, v))

    act_ids = _GRIP_ACT.get(side, [])
    if len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids):
        for aid in act_ids:
            lo, hi = _MODEL.actuator_ctrlrange[aid]
            v = targets[0][1] if targets else _DATA.ctrl[aid]
            v = max(lo, min(hi, v))
            _DATA.ctrl[aid] = v

    controller.update_gripper_hold_position(side)
    _mj_forward()


def set_gripper_force_control(side, delta, controller, force_limit=10.0):
    contact_count = debug_gripper_contacts(side)
    if contact_count > 2 and delta < 0:
        print(f"[{side}] 检测到阻挡，停止闭合")
        return False

    controller.set_gripper_manual_mode(side, True)
    jnames = FINGER_JOINT_NAMES["left"] if side == "left" else FINGER_JOINT_NAMES["right"]
    targets = []

    for jn in jnames:
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        adr = _MODEL.jnt_qposadr[jid]
        lo, hi = _MODEL.jnt_range[jid]

        current_pos = float(_DATA.qpos[adr])
        new_pos = current_pos + float(delta)
        if lo < hi:
            margin = 0.002
            effective_lo = lo + margin
            effective_hi = hi - margin
            new_pos = max(effective_lo, min(effective_hi, new_pos))

        _DATA.qpos[adr] = new_pos
        targets.append((adr, new_pos))

    act_ids = _GRIP_ACT.get(side, [])
    if len(act_ids) == 2 and all(aid is not None and aid >= 0 for aid in act_ids):
        for aid in act_ids:
            lo, hi = _MODEL.actuator_ctrlrange[aid]
            v = targets[0][1] if targets else _DATA.ctrl[aid]
            v = max(lo, min(hi, v))
            _DATA.ctrl[aid] = v

    controller.update_gripper_hold_position(side)
    _mj_forward()
    return True


def print_gripper_state():
    def _read(side):
        jnames = FINGER_JOINT_NAMES["left"] if side == "left" else FINGER_JOINT_NAMES["right"]
        q = []
        for jn in jnames:
            try:
                jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, jn)
                adr = _MODEL.jnt_qposadr[jid]
                q.append(float(_DATA.qpos[adr]))
            except Exception:
                q.append(None)
        a = []
        for aid in _GRIP_ACT.get(side, []):
            a.append(float(_DATA.ctrl[aid]) if (aid is not None and aid >= 0) else None)
        return q, a

    lq, la = _read("left")
    rq, ra = _read("right")
    print(f"[gripper] L qpos={np.round(lq,4)} ctrl={np.round(la,4)} | R qpos={np.round(rq,4)} ctrl={np.round(ra,4)}")


def gripper_troubleshooting():
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


def set_joint_delta(joint_name, delta, controller=None, chains=None):
    try:
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    except Exception:
        jid = -1
    if jid < 0:
        print(f"[WARN] joint not found: {joint_name}")
        return False

    adr = _MODEL.jnt_qposadr[jid]
    lo, hi = _MODEL.jnt_range[jid]
    v = float(_DATA.qpos[adr]) + float(delta)
    if lo < hi:
        v = max(lo, min(hi, v))
    _DATA.qpos[adr] = v

    try:
        trn = _MODEL.actuator_trnid
        for aid in range(_MODEL.nu):
            if trn[aid][0] == jid:
                clo, chi = _MODEL.actuator_ctrlrange[aid]
                tgt = max(clo, min(chi, v))
                _DATA.ctrl[aid] = tgt
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
                    controller.target_qpos[side][idx] = _DATA.qpos[adr]
                    controller.integral_error[side][idx] = 0.0
                    break
        except Exception:
            pass

    mujoco.mj_forward(_MODEL, _DATA)
    return True
