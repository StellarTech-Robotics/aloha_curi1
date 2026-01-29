"""Automation command and trajectory handling for curi1_control."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mujoco

from .pose_utils import quat_normalize, quat_slerp, rpy_to_quat


class AutomationManager:
    """Handles automation command queues and pose trajectories."""

    def __init__(
        self,
        *,
        model,
        data,
        chains,
        controller,
        recorder,
        set_gripper_fn: Callable[[str, float], None],
        set_gripper_force_fn: Callable[[str, float], bool],
        get_targets_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
        set_target_fn: Callable[[str, Optional[np.ndarray], Optional[np.ndarray]], None],
        automation_root: Optional[Path] = None,
        trajectory_time_scale: float = 1.0,
        accel_ratio: float = 0.001,
        decel_ratio: float = 0.001,
    ):
        self.model = model
        self.data = data
        self.chains = chains
        self.controller = controller
        self.recorder = recorder
        self.set_gripper_fn = set_gripper_fn
        self.set_gripper_force_fn = set_gripper_force_fn
        self.get_targets_fn = get_targets_fn
        self.set_target_fn = set_target_fn
        self.time_scale = trajectory_time_scale
        self.accel_ratio = accel_ratio
        self.decel_ratio = decel_ratio

        self.queue: List[Dict] = []
        self.current_cmd: Optional[Dict] = None
        self.wait_until: Optional[float] = None

        root = automation_root or (Path(__file__).resolve().parent.parent / "automation")
        self.automation_dir = Path(root)
        self.automation_dir.mkdir(parents=True, exist_ok=True)
        self.commands_file = self.automation_dir / "commands.json"

        # Trajectory state
        self.trajectory_active = False
        self.trajectory_side: Optional[str] = None
        self.trajectory_positions: List[np.ndarray] = []
        self.trajectory_quaternions: List[np.ndarray] = []
        self.trajectory_index = 0
        self.trajectory_start_time: Optional[float] = None
        self.trajectory_duration = 0.0

    # ----- Trajectory helpers -----
    def _generate_trajectory(self, side: str, target_pos, target_quat, seconds: float, fps: float):
        seconds = max(0.01, seconds)
        steps = max(1, int(seconds * fps))
        c = self.chains[side]
        p0 = self.data.xpos[c["ee"]].copy()
        q0 = self.data.xquat[c["ee"]].copy()

        positions: List[np.ndarray] = []
        quaternions: List[np.ndarray] = []

        accel_ratio = self.accel_ratio
        decel_ratio = self.decel_ratio
        if accel_ratio + decel_ratio > 0.9:
            scale = 0.9 / (accel_ratio + decel_ratio)
            accel_ratio *= scale
            decel_ratio *= scale
        const_ratio = max(0.0, 1.0 - accel_ratio - decel_ratio)
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

    def start_pose_trajectory(self, side: str, target_pos, target_quat, duration, fps=10.0):
        side = side.lower()
        duration_scaled = float(duration) * self.time_scale
        traj = self._generate_trajectory(side, target_pos, target_quat, seconds=duration_scaled, fps=float(fps))
        positions, quats = traj
        if len(positions) == 0:
            return None
        self.trajectory_positions = positions
        self.trajectory_quaternions = quats
        self.trajectory_active = True
        self.trajectory_side = side
        self.trajectory_index = 0
        self.trajectory_start_time = time.time()
        self.trajectory_duration = duration_scaled
        return target_quat

    def stop_trajectory(self) -> bool:
        if not self.trajectory_active:
            return False
        side = self.trajectory_side or "left"
        self.trajectory_active = False
        self.trajectory_positions = []
        self.trajectory_quaternions = []
        self.trajectory_index = 0
        self.trajectory_start_time = None
        current_quat = self.data.xquat[self.chains[side]["ee"]].copy()
        self.set_target_fn(side, quat=current_quat)
        if self.current_cmd and self.current_cmd.get("type") == "move_pose":
            self.current_cmd = None
            print("[automation] move cancelled")
        print("[STOP] 轨迹执行已停止")
        return True

    @property
    def is_trajectory_active(self) -> bool:
        return self.trajectory_active

    def step_trajectory(self, physics_paused: bool):
        if physics_paused or not self.trajectory_active or self.trajectory_side is None:
            return
        side = self.trajectory_side
        if self.trajectory_index < len(self.trajectory_positions):
            target_pos = self.trajectory_positions[self.trajectory_index]
            target_quat = self.trajectory_quaternions[self.trajectory_index]
            self.controller.set_target_from_ik_pose(
                side,
                target_pos,
                target_quat,
                iters=10,
                damping=1e-4,
            )
            self.set_target_fn(side, quat=target_quat.copy())
            self.trajectory_index += 1
            if self.trajectory_index % 10 == 0:
                progress = (self.trajectory_index / len(self.trajectory_positions)) * 100
                current_pos = self.data.xpos[self.controller.chains[side]["ee"]].copy()
                distance = np.linalg.norm(target_pos - current_pos)
                print(f"[轨迹] 进度: {progress:.1f}%, 当前误差: {distance:.4f}m")
            return

        self.trajectory_active = False
        elapsed = 0.0
        if self.trajectory_start_time is not None:
            elapsed = time.time() - self.trajectory_start_time
        print(f"[POSE-IK] 轨迹执行完成，耗时: {elapsed:.2f}秒")
        if self.trajectory_positions and self.trajectory_quaternions:
            final_pos = self.trajectory_positions[-1].copy()
            final_quat = self.trajectory_quaternions[-1].copy()
        else:
            final_pos = self.data.xpos[self.chains[side]["ee"]].copy()
            final_quat = self.data.xquat[self.chains[side]["ee"]].copy()
        self.set_target_fn(side, pos=final_pos, quat=final_quat)
        if self.current_cmd and self.current_cmd.get("type") == "move_pose":
            print("[automation] move_pose completed")
            self.current_cmd = None

    # ----- Command queue handling -----
    def poll_command_file(self):
        if not self.commands_file.exists():
            return
        try:
            with open(self.commands_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                commands = payload.get("commands", [])
            elif isinstance(payload, list):
                commands = payload
            else:
                commands = []
            if commands:
                self.queue.extend(commands)
                print(f"[automation] queued {len(commands)} commands")
        except Exception as exc:
            print(f"[automation] failed to load commands: {exc}")
        finally:
            try:
                self.commands_file.unlink()
            except OSError:
                pass

    def cancel_move_command(self):
        if self.current_cmd and self.current_cmd.get("type") == "move_pose":
            self.current_cmd = None

    def process_commands(self):
        self.poll_command_file()

        if self.current_cmd and self.current_cmd.get("type") == "sleep":
            if self.wait_until is not None and time.time() >= self.wait_until:
                print("[automation] sleep completed")
                self.current_cmd = None
                self.wait_until = None

        while self.current_cmd is None and self.queue:
            cmd_peek = self.queue[0]
            ctype = (cmd_peek.get("type") or "").lower()
            if ctype in {"record", "record_start", "record_stop"}:
                self._handle_record_command(self.queue.pop(0), ctype)
                continue
            if ctype == "gripper":
                cmd = self.queue.pop(0)
                self._handle_gripper_command(cmd)
                continue
            if ctype == "move_pose":
                if self.trajectory_active:
                    break
                cmd = self.queue.pop(0)
                if self._handle_move_pose(cmd):
                    break
                continue
            if ctype == "sleep":
                cmd = self.queue.pop(0)
                seconds = max(0.0, float(cmd.get("seconds", 0.0)))
                self.wait_until = time.time() + seconds
                self.current_cmd = cmd
                print(f"[automation] sleep {seconds:.2f}s")
                break
            if ctype == "gripper_force_lock":
                cmd = self.queue.pop(0)
                self._handle_force_lock(cmd)
                continue
            cmd = self.queue.pop(0)
            print(f"[automation] unknown command type: {ctype}")

    def _handle_record_command(self, cmd: Dict, raw_type: str):
        action = (cmd.get("action") or raw_type.split("_")[-1]).lower()
        try:
            if action in {"start", "on"}:
                if not self.recorder.enabled:
                    L_tgt, R_tgt = self.get_targets_fn()
                    self.recorder.start(self.chains, L_tgt, R_tgt)
                    print("[automation] recording START")
                else:
                    print("[automation] recording already running")
            elif action in {"stop", "off"}:
                if self.recorder.enabled:
                    self.recorder.stop()
                    print("[automation] recording STOP")
                else:
                    print("[automation] recording already stopped")
            else:
                print(f"[automation] unknown record action: {action}")
        except Exception as exc:
            print(f"[automation] record failed: {exc}")

    def _handle_gripper_command(self, cmd: Dict):
        side = cmd.get("side", "right").lower()
        target = cmd.get("target", "open")
        value = cmd.get("value", None)
        duration = float(cmd.get("duration", 0.0))
        delta = float(cmd.get("delta", 0.003))
        step_size = float(cmd.get("step_size", 0.002))
        step_sleep = float(cmd.get("step_sleep", 0.2))
        if side not in ("left", "right"):
            print(f"[automation] invalid gripper side: {side}")
            return
        try:
            # handle textual targets
            if isinstance(target, str):
                t_lower = target.lower()
                if t_lower in {"open", "release"}:
                    self._handle_gripper_open(side, delta, duration, step_size, step_sleep)
                    return
                if t_lower in {"close", "grip"}:
                    self._handle_gripper_close(side, delta, duration, step_size, step_sleep)
                    return
                if t_lower in {"stop", "hold", "hold_position"}:
                    self.controller.set_gripper_manual_mode(side, True)
                    self.controller.update_gripper_hold_position(side)
                    print(f"[automation] gripper {side} -> {target}")
                    return
                print(f"[automation] unknown gripper target: {target}")
                return

            numeric = None
            try:
                numeric = float(target)
            except Exception:
                if value is not None:
                    numeric = float(value)
            if numeric is None:
                print(f"[automation] gripper target invalid: {target}")
                return
            self._handle_gripper_numeric(side, numeric, step_size, step_sleep, cmd)
        except Exception as exc:
            print(f"[automation] gripper failed: {exc}")

    def _current_gripper_value(self, side: str):
        jnames = ["Joint_finger1", "Joint_finger2"] if side == "left" else [
            "r_Joint_finger1",
            "r_Joint_finger2",
        ]
        readings = []
        for jn in jnames:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            except Exception:
                jid = -1
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            readings.append(float(self.data.qpos[adr]))
        if readings:
            return float(np.mean(readings))
        return 0.0

    def _handle_gripper_open(self, side, delta, duration, step_size, step_sleep):
        if duration > 0 or step_sleep > 0:
            current = self._current_gripper_value(side)
            target_pos = max(0.0, current + abs(delta))
            if duration > 0:
                steps = max(1, int(duration / step_sleep)) if step_sleep > 0 else 1
            else:
                steps = max(1, int(abs(delta) / step_size))
            intermediate = np.linspace(current, target_pos, steps + 1)[1:]
            new_cmds = []
            for val in intermediate:
                new_cmds.append({"type": "gripper", "side": side, "target": float(val), "split": True})
                if step_sleep > 0:
                    new_cmds.append({"type": "sleep", "seconds": step_sleep})
            self.queue = new_cmds + self.queue
            print(f"[automation] gripper {side} -> open (gradual, {steps} steps)")
            return
        self.set_gripper_fn(side, abs(delta))
        print(f"[automation] gripper {side} -> open (instant)")

    def _handle_gripper_close(self, side, delta, duration, step_size, step_sleep):
        if duration > 0 or step_sleep > 0:
            current = self._current_gripper_value(side)
            target_pos = max(0.0, current - abs(delta))
            if duration > 0:
                steps = max(1, int(duration / step_sleep)) if step_sleep > 0 else 1
            else:
                steps = max(1, int(abs(delta) / step_size))
            intermediate = np.linspace(current, target_pos, steps + 1)[1:]
            new_cmds = []
            for val in intermediate:
                new_cmds.append({"type": "gripper", "side": side, "target": float(val), "split": True})
                if step_sleep > 0:
                    new_cmds.append({"type": "sleep", "seconds": step_sleep})
            self.queue = new_cmds + self.queue
            print(f"[automation] gripper {side} -> close (gradual, {steps} steps)")
            return
        self.set_gripper_fn(side, -abs(delta))
        print(f"[automation] gripper {side} -> close (instant)")

    def _handle_gripper_numeric(self, side, numeric, step_size, step_sleep, cmd):
        current = self._current_gripper_value(side)
        diff = numeric - current
        if abs(diff) < 1e-4:
            print(f"[automation] gripper {side} already at target {numeric:.3f}")
            return
        split = bool(cmd.get("split", False))
        if not split:
            step_size = max(step_size, 1e-4)
            steps = max(1, int(abs(diff) / step_size))
            if steps > 1:
                intermediate = np.linspace(current, numeric, steps + 1)[1:]
                new_cmds = []
                for val in intermediate:
                    new_cmds.append(
                        {
                            "type": "gripper",
                            "side": side,
                            "target": float(val),
                            "split": True,
                            "step_size": step_size,
                            "step_sleep": step_sleep,
                        }
                    )
                    if step_sleep > 0:
                        new_cmds.append({"type": "sleep", "seconds": step_sleep})
                self.queue = new_cmds + self.queue
                return
        self.controller.set_gripper_manual_mode(side, True)
        jnames = ["Joint_finger1", "Joint_finger2"] if side == "left" else ["r_Joint_finger1", "r_Joint_finger2"]
        for jn in jnames:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            except Exception:
                jid = -1
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            val = max(lo, min(hi, numeric))
            self.data.qpos[adr] = val
        self.controller.update_gripper_hold_position(side)
        mujoco.mj_forward(self.model, self.data)
        print(f"[automation] gripper {side} -> {numeric}")

    def _handle_move_pose(self, cmd: Dict) -> bool:
        side = cmd.get("side", "left").lower()
        try:
            pos = np.array(cmd["pos"], dtype=np.float64)
        except Exception:
            pos = self.data.xpos[self.chains[side]["ee"]].copy()
        if "quat" in cmd:
            quat = quat_normalize(np.array(cmd["quat"], dtype=np.float64))
        elif "rpy_deg" in cmd:
            rpy = np.deg2rad(np.array(cmd["rpy_deg"], dtype=np.float64))
            quat = rpy_to_quat(*rpy)
        elif "rpy_rad" in cmd:
            quat = rpy_to_quat(*cmd["rpy_rad"])
        else:
            quat = self.data.xquat[self.chains[side]["ee"]].copy()
        duration = float(cmd.get("duration", 2.0))
        fps_cmd = float(cmd.get("fps", 60.0))
        final_quat = self.start_pose_trajectory(side, pos, quat, duration=duration, fps=fps_cmd)
        if final_quat is None:
            print(f"[automation] move_pose failed (side={side})")
            return False
        self.set_target_fn(side, quat=final_quat)
        self.current_cmd = cmd
        effective_duration = duration * self.time_scale
        print(f"[automation] move_pose -> {side} pos={np.round(pos,4)} duration={effective_duration:.2f}s (scaled)")
        return True

    def _handle_force_lock(self, cmd: Dict):
        side = cmd.get("side", "right").lower()
        mode = cmd.get("mode", "toggle").lower()
        force = float(cmd.get("force", 10.0))
        if side not in ("left", "right"):
            print(f"[automation] invalid gripper side: {side}")
            return
        if mode == "enable":
            if not self.controller.gripper_force_lock[side]:
                self.controller.toggle_gripper_force_lock(side, force_newton=force)
            else:
                self.controller.gripper_force_target[side] = abs(force)
            print(f"[automation] gripper_force_lock {side} ENABLE ({force:.2f}N)")
        elif mode == "disable":
            if self.controller.gripper_force_lock[side]:
                self.controller.toggle_gripper_force_lock(side, force_newton=force)
            print(f"[automation] gripper_force_lock {side} DISABLE")
        else:
            locked = self.controller.toggle_gripper_force_lock(side, force_newton=force)
            state = "ENABLE" if locked else "DISABLE"
            print(f"[automation] gripper_force_lock {side} {state} ({force:.2f}N)")
