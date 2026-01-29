"""Pose and quaternion helper utilities for curi1_control."""
from __future__ import annotations
import numpy as np

_DATA = None


def configure(data):
    global _DATA
    _DATA = data


def quat_to_rpy(qw, qx, qy, qz):
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw*qy - qz*qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def get_ee_pose(controller, side):
    ee = controller.chains[side]["ee"]
    pos = _DATA.xpos[ee].copy()
    quat = _DATA.xquat[ee].copy()
    return pos, quat


def print_ee_pose(controller, side=None):
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
        except Exception as exc:
            print(f"[{s}] 读取EE位姿失败: {exc}")


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
    qd = quat_normalize(q_des)
    qc = quat_normalize(q_cur)
    dq = quat_mul(qd, quat_conj(qc))
    if dq[0] < 0:
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
    cr = np.cos(roll*0.5); sr = np.sin(roll*0.5)
    cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
    cy = np.cos(yaw*0.5); sy = np.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return quat_normalize(np.array([w,x,y,z], dtype=np.float64))
