#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成抓取/放置演示动作队列，供 curi1_control.py 自动执行。

步骤：
1. 在终端 A 中运行  并保持运行。
2. 在终端 B 中执行本脚本 。
   脚本会写入 automation/commands.json，控制程序检测到后自动依次执行：
     - 开始录制
     - 一系列右臂姿态轨迹
     - 停止录制
"""

import json
import time
from pathlib import Path

import mujoco
import numpy as np

OPEN_SEQUENCE = [0.001, 0.035]
CLOSE_SEQUENCE = [0.0335, 0.0325, 0.0315, 0.0305, 0.0295, 0.0285, 0.026, 0.023, 0.020, 0.019]
RELEASE_SEQUENCE = [0.020, 0.024, 0.027, 0.031, 0.035]
RELEASE_CLOSE_SEQUENCE = [0.0335,  0.0]

def move(side, pos, rpy_deg, duration=2.0):
    return {
        "type": "move_pose",
        "side": side,
        "pos": pos,
        "rpy_deg": rpy_deg,
        "duration": duration,
        "fps": 60.0,
    }


def main():
    root = Path(__file__).resolve().parent
    automation_file = root / "commands.json"

    pose_file = root / "box_pose.json"
    if pose_file.exists():
        try:
            payload = json.loads(pose_file.read_text())
            box_pos = np.array(payload.get("box_pos", [-0.295, -0.520, 0.57]), dtype=np.float64)
        except Exception:
            box_pos = np.array([-0.295, -0.520, 0.57], dtype=np.float64)
    else:
        model_path = (root.parent / "assets" / "bimanual_curi1_transfer_cube.xml").resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_pos = data.xpos[box_body].copy()

    def move_with_settle(*args, settle=0.5, **kwargs):
        cmds = [move(*args, **kwargs)]
        if settle > 0:
            cmds.append({"type": "sleep", "seconds": float(settle)})
        return cmds

    def gripper_sequence(side, targets, total_time=3.0, step=5e-4):
        cmds = []
        if not targets:
            return cmds
        expanded = []
        prev = None
        for val in targets:
            val = float(val)
            if prev is None:
                expanded.append(val)
            else:
                diff = val - prev
                steps = max(1, int(abs(diff) / max(step, 1e-6)))
                interp = np.linspace(prev, val, steps + 1)[1:]
                expanded.extend([float(v) for v in interp])
            prev = val
        step_time = total_time / len(expanded) if total_time > 0 and expanded else 0.0
        for val in expanded:
            cmds.append({
                "type": "gripper",
                "side": side,
                "target": val,
                "step_size": step,
                "step_sleep": 0.0,
            })
            if step_time > 0:
                cmds.append({"type": "sleep", "seconds": step_time})
        return cmds

    num_runs = 1 # run 1 times
    sequence = []
    for run_idx in range(num_runs):
        ee_x = float(box_pos[0] + 0.005)
        ee_y = float(box_pos[1] + 0.18)
        sequence.extend([
            {"type": "record", "action": "start"},
            {"type": "sleep", "seconds": 0.2},
        ])
        sequence += move_with_settle("right", [ee_x, ee_y, 0.62], [90, 0, 0], duration=1.0, settle=0.5)
        sequence += move_with_settle("right", [ee_x, ee_y, 0.62], [105, 0, 0], duration=1.0, settle=0.5)
        sequence += gripper_sequence("right", OPEN_SEQUENCE, total_time=0.6)
        sequence += move_with_settle("right", [ee_x, ee_y-0.05, 0.59], [105, 0, 0], duration=1.0, settle=0.5)
        sequence += gripper_sequence("right", CLOSE_SEQUENCE, total_time=0.6)
        sequence.append({"type": "gripper_force_lock", "side": "right", "force": 15.0, "mode": "enable"})
        sequence += move_with_settle("right", [ee_x, ee_y-0.05, 0.70], [105, 0, 0], duration=1.0, settle=0.5)
        sequence += move_with_settle("right", [0.0, ee_y-0.05, 0.70], [105, 0, 0], duration=1.0, settle=0.5)
        sequence += move_with_settle("right", [0.0, ee_y-0.05, 0.59], [105, 0, 0], duration=1.0, settle=0.5)
        sequence.append({"type": "gripper_force_lock", "side": "right", "mode": "disable"})
        sequence += gripper_sequence("right", RELEASE_SEQUENCE, total_time=0.6)
        sequence += move_with_settle("right", [0.0, ee_y-0.05, 0.70], [105, 0, 0], duration=1.0, settle=0.5)
        sequence += gripper_sequence("right", RELEASE_CLOSE_SEQUENCE, total_time=0.6)
        sequence.append({"type": "gripper_force_lock", "side": "right", "mode": "disable"})
        sequence += move_with_settle("right", [-0.4215, -0.5276, 0.6486], [93.2, -8.9, 2.0], duration=1.0, settle=0.5)
        sequence.append({"type": "record", "action": "stop"})
        if run_idx != num_runs - 1:
            sequence.append({"type": "sleep", "seconds": 1.0})

    automation_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "commands": sequence,
    }
    automation_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[automation] 写入 {automation_file}，共 {len(sequence)} 条指令")
    print("[automation] 请确认 curi1_control.py 正在运行，机械人将自动执行动作并录制视频。")


if __name__ == "__main__":
    main()
