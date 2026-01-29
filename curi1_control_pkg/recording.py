"""Recording and preview utilities for curi1_control."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np


class _VideoSink:
    """Thin wrapper around cv2/imageio writers."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.kind = None
        self._warned = False
        self._writer = None

        try:
            import cv2  # type: ignore

            fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
            writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
            if writer is not None and writer.isOpened():
                self._writer = writer
                self.kind = "cv2"
            else:
                if writer is not None:
                    writer.release()
        except Exception:
            self._writer = None

        if self._writer is None:
            try:
                import imageio  # type: ignore

                self._imageio = imageio
                self._writer = imageio.get_writer(self.path, fps=self.fps)
                self.kind = "iio"
            except Exception as exc:
                print("[record] ERROR: no video backend (cv2 or imageio).", exc)
                self.kind = None
                self._writer = None

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._writer is None:
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
                frame = frame[: self.height, : self.width]

        if self.kind == "cv2":
            import cv2  # type: ignore

            if hasattr(self._writer, "isOpened") and not self._writer.isOpened():
                if not self._warned:
                    print("[record] WARNING: cv2 writer closed, stop writing")
                    self._warned = True
                return
            bgr = frame[..., ::-1].copy()
            self._writer.write(bgr)
        elif self.kind == "iio":
            try:
                self._writer.append_data(frame)
            except AttributeError:
                self._writer.write(frame)

    def close(self) -> None:
        if self._writer is None:
            return
        if self.kind == "cv2":
            self._writer.release()
        elif self.kind == "iio":
            self._writer.close()
        self._writer = None


class Recorder:
    """Handles multi-camera recording + dataset dumps."""

    def __init__(
        self,
        model,
        data,
        *,
        cams: Sequence[str] = ("top", "left_wrist", "right_wrist"),
        fps: float = 15,
        size: Tuple[int, int] = (640, 480),
        out_root: Optional[str] = None,
        qpos_extractor: Optional[Callable[[np.ndarray, Dict], np.ndarray]] = None,
    ):
        self.model = model
        self.data = data
        self.cams = list(cams)
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        if out_root is None:
            default_root = Path(__file__).resolve().parent.parent / "datasets"
        else:
            default_root = Path(out_root)
        self.out_root = str(default_root)
        self.renderers = {}
        self.videos: Dict[str, _VideoSink] = {}
        self.csv = None
        self.writer: Optional[csv.writer] = None
        self.csv_path = None
        self.meta = {}
        self.t0 = None
        self.last = 0.0
        self.interval = 1.0 / self.fps
        self.frame_id = 0
        self.enabled = False
        self.have_h5py = False
        self.h5_path = None

        self.action_buffer: List[np.ndarray] = []
        self.qpos_buffer: List[np.ndarray] = []
        self.qvel_buffer: List[np.ndarray] = []
        self.image_buffers: Dict[str, List[np.ndarray]] = {cam: [] for cam in cams}
        self.combo_video: Optional[_VideoSink] = None
        self.composite_order = ("left_wrist", "right_wrist", "top")
        self._combo_warned = False
        self._plot_warned = False
        self.qpos_extractor = qpos_extractor or (lambda arr, _chains: np.asarray(arr, dtype=np.float32))

    def _ensure_renderers(self) -> None:
        max_w = int(getattr(self.model.vis.global_, "offwidth", 320))
        max_h = int(getattr(self.model.vis.global_, "offheight", 240))
        req_w, req_h = int(self.size[0]), int(self.size[1])
        width = min(req_w, max_w) if max_w > 0 else req_w
        height = min(req_h, max_h) if max_h > 0 else req_h
        for name in self.cams:
            if name in self.renderers:
                continue
            ok = False
            try:
                self.renderers[name] = mujoco.Renderer(self.model, width, height)
                ok = True
                print(f"[record] renderer[{name}] size = {width}x{height} (offbuffer {max_w}x{max_h})")
            except Exception as exc:
                try:
                    self.renderers[name] = mujoco.Renderer(self.model, height, width)
                    ok = True
                    print(f"[record] renderer[{name}] size = {height}x{width} (swapped; offbuffer {max_w}x{max_h})")
                except Exception as exc2:
                    print(f"[record] ERROR creating renderer for {name}: {exc} | swapped: {exc2}")
            if not ok:
                raise RuntimeError(f"Cannot create offscreen renderer for camera {name}")

    def start(self, chains, L_tgt, R_tgt) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = Path(self.out_root) / f"run_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)
        self._ensure_renderers()

        self.action_buffer = []
        self.qpos_buffer = []
        self.qvel_buffer = []
        self.image_buffers = {cam: [] for cam in self.cams}

        for cam in self.cams:
            sink = _VideoSink(str(outdir / f"{cam}.mp4"), self.size[0], self.size[1], self.fps)
            self.videos[cam] = sink

        if all(cam in self.cams for cam in self.composite_order):
            combo_name = "_".join(self.composite_order) + "_combo.mp4"
            self.combo_video = _VideoSink(
                str(outdir / combo_name),
                self.size[0] * len(self.composite_order),
                self.size[1],
                self.fps,
            )
        else:
            self.combo_video = None
        self._combo_warned = False
        self._plot_warned = False

        self.csv_path = str(outdir / "states.csv")
        self.csv = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv)

        mobile_names = [
            "l_joint1",
            "l_joint2",
            "l_joint3",
            "l_joint4",
            "l_joint5",
            "l_joint6",
            "l_gripper",
            "r_joint1",
            "r_joint2",
            "r_joint3",
            "r_joint4",
            "r_joint5",
            "r_joint6",
            "r_gripper",
        ]
        header = (
            ["t", "frame"]
            + mobile_names
            + [
                "Lx",
                "Ly",
                "Lz",
                "Lqw",
                "Lqx",
                "Lqy",
                "Lqz",
                "Rx",
                "Ry",
                "Rz",
                "Rqw",
                "Rqx",
                "Rqy",
                "Rqz",
            ]
            + ["L_tgt_x", "L_tgt_y", "L_tgt_z", "R_tgt_x", "R_tgt_y", "R_tgt_z"]
        )
        self.writer.writerow(header)

        self.meta = {
            "fps": self.fps,
            "size": self.size,
            "cameras": self.cams,
            "original_nq": int(self.model.nq),
            "original_nv": int(self.model.nv),
            "mobile_aloha_qpos_dim": 14,
            "mobile_aloha_format": "l_joint1-6, l_gripper, r_joint1-6, r_gripper",
            "gripper_mapping": "CURI双手指夹爪映射为单维夹爪 (取平均值)",
            "bodies": {
                "left_ee": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, chains["left"]["ee"]),
                "right_ee": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, chains["right"]["ee"]),
            },
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

        try:
            import h5py  # type: ignore

            self.have_h5py = True
            self.h5_path = str(outdir / "episode_0.hdf5")
            self._h5py = h5py
        except Exception as exc:
            self.have_h5py = False
            self._h5py = None
            print("[record] h5py not available, will skip episode.hdf5:", exc)

        self.outdir = str(outdir)
        self.t0 = time.time()
        self.last = 0.0
        self.frame_id = 0
        self.enabled = True
        print(f"[record] START -> {os.path.abspath(self.outdir)}")

    def stop(self) -> None:
        if self.csv:
            self.csv.close()
            self.csv = None
            self.writer = None
        for sink in self.videos.values():
            sink.close()
        self.videos.clear()
        if self.combo_video is not None:
            self.combo_video.close()
            self.combo_video = None

        if self.have_h5py and self.h5_path is not None and self.qpos_buffer:
            self._save_hdf5()
        print(f"[record] STOP  -> {os.path.abspath(self.outdir)}")
        self._save_qpos_plot()
        self.enabled = False

    def _save_hdf5(self) -> None:
        if not self.have_h5py or self._h5py is None or self.h5_path is None:
            return
        try:
            with self._h5py.File(self.h5_path, "w") as h5:
                obs_grp = h5.create_group("observations")
                img_grp = obs_grp.create_group("images")

                cam_map = {
                    "left_wrist": "cam_left_wrist",
                    "right_wrist": "cam_right_wrist",
                    "top": "cam_high",
                }
                for cam_orig, cam_name in cam_map.items():
                    if cam_orig in self.image_buffers and self.image_buffers[cam_orig]:
                        img_data = np.array(self.image_buffers[cam_orig], dtype=np.uint8)
                        img_grp.create_dataset(
                            cam_name, data=img_data, compression="gzip", compression_opts=4
                        )
                        print(f"  - images/{cam_name}: {img_data.shape}")

                qpos_data = None
                if self.qpos_buffer:
                    qpos_data = np.array(self.qpos_buffer, dtype=np.float32)
                    obs_grp.create_dataset("qpos", data=qpos_data, compression="gzip", compression_opts=4)
                    print(f"  - qpos: {qpos_data.shape}")
                elif self.action_buffer:
                    qpos_data = np.array(self.action_buffer, dtype=np.float32)
                    obs_grp.create_dataset("qpos", data=qpos_data, compression="gzip", compression_opts=4)
                    print(f"  - qpos(fallback action): {qpos_data.shape}")

                qvel_data = None
                if self.qvel_buffer:
                    qvel_data = np.array(self.qvel_buffer, dtype=np.float32)
                    obs_grp.create_dataset("qvel", data=qvel_data, compression="gzip", compression_opts=4)
                    print(f"  - qvel: {qvel_data.shape}")

                action_data = None
                if self.action_buffer:
                    action_data = np.array(self.action_buffer, dtype=np.float32)
                    h5.create_dataset("action", data=action_data, compression="gzip", compression_opts=4)
                    print(f"  - action: {action_data.shape}")

                if qpos_data is not None:
                    h5.create_dataset("/qpos", data=qpos_data, compression="gzip", compression_opts=4)
                if qvel_data is not None:
                    h5.create_dataset("/qvel", data=qvel_data, compression="gzip", compression_opts=4)
                if action_data is not None:
                    h5.create_dataset("/action", data=action_data, compression="gzip", compression_opts=4)

                h5.attrs["sim"] = True
                h5.attrs["num_timesteps"] = len(self.qpos_buffer)
                h5.attrs["fps"] = self.fps
            print(f"[record] episode_0.hdf5 saved at {os.path.abspath(self.h5_path)}")
        except Exception as exc:
            print(f"[record] Failed to save HDF5: {exc}")

    def step(self, chains, L_tgt, R_tgt, controller=None) -> None:
        if not self.enabled or self.t0 is None:
            return
        t_rel = time.time() - self.t0
        if t_rel - self.last + 1e-9 < self.interval:
            return
        self.last += self.interval

        frame_cache = {}
        for cam, renderer in self.renderers.items():
            try:
                renderer.update_scene(self.data, camera=cam)
                rgb = renderer.render()
                frame_cache[cam] = rgb
                if cam in self.videos:
                    self.videos[cam].write(rgb)
                self.image_buffers[cam].append(rgb.copy())
            except Exception as exc:
                print(f"[record] camera {cam}: {exc}")

        if self.combo_video is not None:
            if all(cam in frame_cache for cam in self.composite_order):
                combo = np.concatenate([frame_cache[cam] for cam in self.composite_order], axis=1)
                self.combo_video.write(combo)
            else:
                if not self._combo_warned:
                    print("[record] 缺少三路画面，无法生成合成视频")
                    self._combo_warned = True

        qpos_14dim = self.qpos_extractor(self.data.qpos, chains)
        self.qpos_buffer.append(qpos_14dim.copy())
        self.qvel_buffer.append(self.data.qvel.copy())
        if controller is not None:
            self.action_buffer.append(
                np.concatenate([controller.target_qpos["left"], controller.target_qpos["right"]])
            )

        if self.csv and self.writer:
            Lp = self.data.xpos[chains["left"]["ee"]].copy()
            Lq = self.data.xquat[chains["left"]["ee"]].copy()
            Rp = self.data.xpos[chains["right"]["ee"]].copy()
            Rq = self.data.xquat[chains["right"]["ee"]].copy()
            row = [
                t_rel,
                self.frame_id,
                *qpos_14dim.tolist(),
                *Lp.tolist(),
                *Lq.tolist(),
                *Rp.tolist(),
                *Rq.tolist(),
                float(L_tgt[0]),
                float(L_tgt[1]),
                float(L_tgt[2]),
                float(R_tgt[0]),
                float(R_tgt[1]),
                float(R_tgt[2]),
            ]
            try:
                self.writer.writerow(row)
            except Exception:
                pass

        self.frame_id += 1

    def _save_qpos_plot(self) -> None:
        if not self.qpos_buffer and self.action_buffer:
            self.qpos_buffer = [np.asarray(cmd, dtype=np.float32) for cmd in self.action_buffer]
        if not self.qpos_buffer:
            if not self._plot_warned:
                print("[record] skip qpos plot (no frames captured)")
                self._plot_warned = True
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            if not self._plot_warned:
                print(f"[record] matplotlib not available, skip qpos plot: {exc}")
                self._plot_warned = True
            return
        try:
            data = np.array(self.qpos_buffer, dtype=np.float32)
            steps = data.shape[0]
            t = np.arange(steps)
            out_path = Path(self.outdir) / "episode_0_qpos.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            cmap = plt.get_cmap("tab10")

            left_names = [
                "L_joint0",
                "L_joint1",
                "L_joint2",
                "L_joint3",
                "L_joint4",
                "L_joint5",
                "L_gripper",
            ]
            for i, (idx, name) in enumerate(zip(range(7), left_names)):
                axes[0].plot(t, data[:, idx], label=name, color=cmap(i % 10))
            axes[0].set_ylabel("Left arm / gripper (Mobile ALOHA)")
            axes[0].legend(loc="upper right", fontsize=8, ncol=3)
            axes[0].grid(True, linestyle="--", alpha=0.3)

            right_names = [
                "R_joint0",
                "R_joint1",
                "R_joint2",
                "R_joint3",
                "R_joint4",
                "R_joint5",
                "R_gripper",
            ]
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
        except Exception as exc:
            print(f"[record] Failed to save qpos plot: {exc}")


class CamPreview:
    """Simple OpenCV-based camera preview helper."""

    def __init__(
        self,
        model,
        data,
        *,
        cams: Sequence[str] = ("top", "left_wrist", "right_wrist"),
        size: Tuple[int, int] = (320, 240),
        fps: float = 5,
        window: str = "cams",
        suppress_qt_warnings: Optional[Callable[[], None]] = None,
    ):
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
        self._suppress_qt_warnings = suppress_qt_warnings

        try:
            import cv2 as _cv2

            self._cv2 = _cv2
            if self._suppress_qt_warnings:
                self._suppress_qt_warnings()
            print("[preview] OpenCV GUI 测试已跳过（WSL环境）")
            print("[preview] 摄像头预览功能将被禁用，但录制功能仍然可用")
        except Exception as exc:
            print("[preview] OpenCV not available:", exc, "-> run: pip install opencv-python")

        for name in self.cams:
            print(f"[preview] 正在创建 renderer[{name}]...")
            try:
                self.renderers[name] = mujoco.Renderer(model, self.size[0], self.size[1])
                print(f"[preview] renderer[{name}] size = {self.size[0]}x{self.size[1]}")
            except Exception as exc:
                print(f"[preview] 第一次尝试失败: {exc}")
                try:
                    self.renderers[name] = mujoco.Renderer(model, self.size[1], self.size[0])
                    print(f"[preview] renderer[{name}] size = {self.size[1]}x{self.size[0]} (swapped)")
                except Exception as exc2:
                    print(f"[preview] ERROR creating renderer for {name}: {exc} | swapped: {exc2}")
        print("[preview] 所有渲染器创建完成")

    def toggle(self) -> None:
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

    def step(self, t_now: float) -> None:
        if not self.enabled or self._cv2 is None:
            return
        if t_now - self.last_t + 1e-9 < self.interval:
            return
        self.last_t += self.interval

        try:
            frames = []
            for cam, renderer in self.renderers.items():
                try:
                    renderer.update_scene(self.data, camera=cam)
                    rgb = renderer.render()
                    bgr = rgb[..., ::-1].copy()
                    self._cv2.putText(
                        bgr,
                        cam,
                        (8, 20),
                        self._cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        self._cv2.LINE_AA,
                    )
                    frames.append(bgr)
                except Exception as exc:
                    print(f"[preview] {cam}: {exc}")
            if not frames:
                return
            try:
                panel = self._cv2.hconcat(frames)
            except Exception:
                height = min(f.shape[0] for f in frames)
                width = min(f.shape[1] for f in frames)
                frames = [f[:height, :width] for f in frames]
                panel = self._cv2.hconcat(frames)

            self._cv2.imshow(self.window, panel)
            self._cv2_window_created = True
            self._cv2.waitKey(1)
        except Exception as exc:
            if not self._qt_error_shown:
                print(f"[preview] GUI错误 (Qt线程问题): {exc}")
                print("[preview] 自动禁用预览功能。录制功能不受影响。")
                self._qt_error_shown = True
            self.enabled = False
            self._cv2 = None
