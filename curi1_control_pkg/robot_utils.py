"""Robot-related helper utilities."""
from __future__ import annotations
import numpy as np
import mujoco

_MODEL = None
_DATA = None

LEFT_JOINT_PREFIX = "l_joint"
RIGHT_JOINT_PREFIX = "r_joint"
LEFT_BASE_BODY = "l_base_link1"
RIGHT_BASE_BODY = "r_base_link1"
LEFT_EE_BODY = "l_rmg42_base_link"
RIGHT_EE_BODY = "r_rmg42_base_link"


def configure(model, data):
    global _MODEL, _DATA
    _MODEL = model
    _DATA = data


def mj_id(objtype, name):
    try:
        return mujoco.mj_name2id(_MODEL, objtype, name)
    except Exception:
        return -1


def find_arm_chain():
    left, right = [], []
    for j in range(_MODEL.njnt):
        name = mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.startswith(LEFT_JOINT_PREFIX):
            left.append((j, name))
        elif name.startswith(RIGHT_JOINT_PREFIX):
            right.append((j, name))

    def sort6(pairs):
        def k(p):
            j, name = p
            num = "".join([c for c in name if c.isdigit()])
            return int(num) if num else 999
        return [j for j, _ in sorted(pairs, key=k)][:6]

    l_ids = sort6(left)
    r_ids = sort6(right)
    l_qadr = [ _MODEL.jnt_qposadr[j] for j in l_ids]
    r_qadr = [ _MODEL.jnt_qposadr[j] for j in r_ids]
    return {
        "left":  {"base": mj_id(mujoco.mjtObj.mjOBJ_BODY, LEFT_BASE_BODY),  "ee": mj_id(mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY),  "jids": l_ids, "qadr": l_qadr},
        "right": {"base": mj_id(mujoco.mjtObj.mjOBJ_BODY, RIGHT_BASE_BODY), "ee": mj_id(mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY), "jids": r_ids, "qadr": r_qadr},
    }


def print_chain(chains):
    print("=== Arm chain discovery ===")
    for side in ("left", "right"):
        c = chains[side]
        jnames = [mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, j) for j in c["jids"]]
        print(f"{side.upper()}: base={mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_BODY, c['base'])} -> ee={mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_BODY, c['ee'])}")
        print("  joints:", jnames)


def extract_14dim_qpos(data_qpos, chains):
    result = np.zeros(14, dtype=np.float32)
    left_chain = chains["left"]
    for i, qadr in enumerate(left_chain["qadr"]):
        result[i] = data_qpos[qadr]
    try:
        f1 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger1")
        f2 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger2")
        q1 = _MODEL.jnt_qposadr[f1]
        q2 = _MODEL.jnt_qposadr[f2]
        result[6] = (data_qpos[q1] + data_qpos[q2]) / 2.0
    except Exception:
        result[6] = 0.0
    right_chain = chains["right"]
    for i, qadr in enumerate(right_chain["qadr"]):
        result[7 + i] = data_qpos[qadr]
    try:
        f1 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger1")
        f2 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger2")
        q1 = _MODEL.jnt_qposadr[f1]
        q2 = _MODEL.jnt_qposadr[f2]
        result[13] = (data_qpos[q1] + data_qpos[q2]) / 2.0
    except Exception:
        result[13] = 0.0
    return result


def expand_14dim_to_full_qpos(mobile_qpos, current_full_qpos, chains):
    result = current_full_qpos.copy()
    left_chain = chains["left"]
    for i, qadr in enumerate(left_chain["qadr"]):
        result[qadr] = mobile_qpos[i]
    try:
        f1 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger1")
        f2 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "Joint_finger2")
        q1 = _MODEL.jnt_qposadr[f1]
        q2 = _MODEL.jnt_qposadr[f2]
        gripper_val = mobile_qpos[6]
        result[q1] = gripper_val
        result[q2] = -gripper_val
    except Exception:
        pass
    right_chain = chains["right"]
    for i, qadr in enumerate(right_chain["qadr"]):
        result[qadr] = mobile_qpos[7 + i]
    try:
        f1 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger1")
        f2 = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "r_Joint_finger2")
        q1 = _MODEL.jnt_qposadr[f1]
        q2 = _MODEL.jnt_qposadr[f2]
        gripper_val = mobile_qpos[13]
        result[q1] = gripper_val
        result[q2] = -gripper_val
    except Exception:
        pass
    return result


def find_joint_actuators():
    joint_to_actuator = {}
    for aid in range(_MODEL.nactuator):
        jid = _MODEL.actuator_trnid[aid][0]
        if jid >= 0:
            joint_to_actuator[jid] = aid
    return joint_to_actuator


def find_gripper_actuators():
    names = {
        "left":  ["Joint_finger1", "Joint_finger2"],
        "right": ["r_Joint_finger1", "r_Joint_finger2"],
    }
    out = {"left": [], "right": []}
    for side, lst in names.items():
        ids = []
        for nm in lst:
            try:
                aid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except Exception:
                aid = -1
            ids.append(aid)
        out[side] = ids
    return out


def debug_gripper_contacts(side="left"):
    finger_names = ["Joint_finger1", "Joint_finger2"] if side == "left" else ["r_Joint_finger1", "r_Joint_finger2"]
    print(f"\n=== {side.upper()} 夹爪碰撞检测 ===")
    contact_count = 0
    for i in range(_DATA.ncon):
        con = _DATA.contact[i]
        geom1 = _MODEL.geom_id2name(con.geom1)
        geom2 = _MODEL.geom_id2name(con.geom2)
        is_contact = any(f in geom1 or f in geom2 for f in finger_names)
        if is_contact:
            contact_count += 1
            print(f"  碰撞: {geom1} <-> {geom2}")
    print(f"总碰撞数: {contact_count}\n")
    return contact_count


def check_cube_proximity(side="left", threshold=0.06):
    ee = _DATA.xpos[_MODEL.body(LEFT_EE_BODY if side == "left" else RIGHT_EE_BODY)].copy()
    box_body = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_BODY, "box")
    cube_pos = _DATA.xpos[box_body].copy()
    dist = np.linalg.norm(cube_pos - ee)
    return dist < threshold, dist


def print_enhanced_gripper_state():
    pass


def set_joint_delta(joint_name, delta, controller=None, chains=None):
    try:
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    except Exception:
        jid = -1
    if jid < 0:
        return False
    adr = _MODEL.jnt_qposadr[jid]
    _DATA.qpos[adr] += delta
    mujoco.mj_forward(_MODEL, _DATA)
    if controller is not None:
        if chains and joint_name.startswith('l_'):
            controller.set_target_from_ik_pose('left', _DATA.xpos[chains['left']['ee']], _DATA.xquat[chains['left']['ee']], iters=0)
        elif chains and joint_name.startswith('r_'):
            controller.set_target_from_ik_pose('right', _DATA.xpos[chains['right']['ee']], _DATA.xquat[chains['right']['ee']], iters=0)
    return True
