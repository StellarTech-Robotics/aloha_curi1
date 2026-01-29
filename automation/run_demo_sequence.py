#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利用 curi1_control.py 的自动化指令通道，推送一段固定的机械臂动作。

使用方法：
1. 在一个终端中启动 `python3 curi1_control.py` 并保持运行。
2. 在另一终端执行本脚本 `python3 automation/run_demo_sequence.py`。
   若队列中暂无命令，脚本会生成 automation/commands.json，随后
   curi1_control.py 会按顺序自动执行这些动作。
"""

import json
import time
from pathlib import Path


def build_move(side, pos, rpy_deg=None, quat=None, duration=3.0, fps=60.0):
    """构造 move_pose 指令，默认使用欧拉角(度)描述姿态。"""
    cmd = {
        "type": "move_pose",
        "side": side,
        "pos": pos,
        "duration": duration,
        "fps": fps,
    }
    if quat is not None:
        cmd["quat"] = quat
    elif rpy_deg is not None:
        cmd["rpy_deg"] = rpy_deg
    else:
        cmd["rpy_deg"] = [90.0, 0.0, 0.0]
    return cmd


def main():
    root = Path(__file__).resolve().parent
    automation_file = root / "commands.json"
    sequence = [
        build_move(
            side="right",
            pos=[-0.30, -0.52, 0.62],
            rpy_deg=[90.0, 0.0, 0.0],
            duration=3.0,
        ),
        {"type": "sleep", "seconds": 1.0},
        build_move(
            side="right",
            pos=[-0.25, -0.45, 0.66],
            rpy_deg=[90.0, -15.0, 0.0],
            duration=2.5,
        ),
        {"type": "sleep", "seconds": 1.0},
        build_move(
            side="right",
            pos=[-0.32, -0.55, 0.60],
            rpy_deg=[100.0, 5.0, -10.0],
            duration=3.0,
        ),
        {"type": "sleep", "seconds": 1.5},
        build_move(
            side="left",
            pos=[0.28, -0.52, 0.62],
            rpy_deg=[90.0, 0.0, 0.0],
            duration=3.0,
        ),
    ]

    payload = {
        "timestamp": time.time(),
        "commands": sequence,
    }

    automation_file.parent.mkdir(parents=True, exist_ok=True)
    automation_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[automation] 写入 {automation_file}，共 {len(sequence)} 条指令")
    print("[automation] 请确保 curi1_control.py 正在运行，机械人会自动执行队列中的动作。")


if __name__ == "__main__":
    main()
