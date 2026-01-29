#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mirror test6.py import order exactly
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "bimanual_curi1_transfer_cube.xml"

# ===== MuJoCo setup (same style as test6.py) =====
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

# ===== IK (numeric Jacobian + DLS) =====
def numeric_jac(ch, side, eps=1e-5):
    c = ch[side]
    J = np.zeros((3, len(c["qadr"])))
    qbackup = data.qpos.copy()
    mujoco.mj_forward(model, data)
    for k, adr in enumerate(c["qadr"]):
        data.qpos[:] = qbackup; data.qpos[adr] += eps; mujoco.mj_forward(model, data); p_plus = data.xpos[c["ee"]].copy()
        data.qpos[:] = qbackup; data.qpos[adr] -= eps; mujoco.mj_forward(model, data); p_minus= data.xpos[c["ee"]].copy()
        J[:,k] = (p_plus-p_minus)/(2*eps)
    data.qpos[:] = qbackup; mujoco.mj_forward(model, data)
    return J

def ik_step(ch, side, target, lam=1e-3, dq_max=0.05):
    c = ch[side]
    cur = data.xpos[c["ee"]].copy()
    e = target - cur
    J = numeric_jac(ch, side)
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(3), e)
    dq = np.clip(dq, -dq_max, dq_max)
    for jid, adr, inc in zip(c["jids"], c["qadr"], dq):
        v = data.qpos[adr] + float(inc)
        rmin, rmax = model.jnt_range[jid]
        if rmin < rmax:
            v = max(rmin, min(rmax, v))
        data.qpos[adr] = v
    mujoco.mj_forward(model, data)

# 放在 MuJoCo 初始化后
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

def set_gripper(side, delta):
    """
    同时写 qpos 和 actuator ctrl，立即可见且不会被拉回。
    delta>0 变大（张开），delta<0 变小（闭合）
    """
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

    mujoco.mj_forward(model, data)  # 立即刷新画面

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

    # 同步所有“传动目标是这个关节”的 actuator 的 ctrl（通常是 position actuator）
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


# ================== Recording (multi-camera) ==================
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
        # HDF5
        self.h5 = None
        self.h5_dsets = {}
        self.have_h5py = False

    def _ensure_renderers(self):
        # Clamp to offscreen framebuffer size to avoid height>framebuffer errors
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
        # videos
        for cam in self.cams:
            sink = _VideoSink(str(outdir / f"{cam}.mp4"), self.size[0], self.size[1], self.fps)
            self.videos[cam] = sink
        # csv
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
        # meta
        self.meta = {
            "fps": self.fps,
            "size": self.size,
            "cameras": self.cams,
            "nq": int(self.model.nq),
            "bodies": {
                "left_ee": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, chains["left"]["ee"]),
                "right_ee": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, chains["right"]["ee"]),
            },
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        # HDF5
        self.h5 = None; self.h5_dsets = {}; self.have_h5py = False
        try:
            import h5py
            self.have_h5py = True
            self.h5 = h5py.File(str(outdir / 'episode.hdf5'), 'w')
            g = self.h5.create_group('images')
            H, W = self.size[1], self.size[0]
            cam_map = {'top':'cam_high','left_wrist':'cam_left_wrist','right_wrist':'cam_right_wrist'}
            for cam in self.cams:
                name = cam_map.get(cam, cam)
                self.h5_dsets[name] = g.create_dataset(name, shape=(0,H,W,3), maxshape=(None,H,W,3),
                                                       dtype='uint8', chunks=(1,H,W,3), compression='gzip', compression_opts=4)
            self.h5_dsets['qpos'] = self.h5.create_dataset('qpos', shape=(0,self.model.nq), maxshape=(None,self.model.nq),
                                                           dtype='float64', chunks=True, compression='gzip', compression_opts=4)
            self.h5_dsets['qvel'] = self.h5.create_dataset('qvel', shape=(0,self.model.nv), maxshape=(None,self.model.nv),
                                                           dtype='float64', chunks=True, compression='gzip', compression_opts=4)
            self.h5_dsets['L_tgt'] = self.h5.create_dataset('L_tgt', shape=(0,3), maxshape=(None,3),
                                                            dtype='float32', chunks=True, compression='gzip', compression_opts=4)
            self.h5_dsets['R_tgt'] = self.h5.create_dataset('R_tgt', shape=(0,3), maxshape=(None,3),
                                                            dtype='float32', chunks=True, compression='gzip', compression_opts=4)
            self.h5_dsets['frame'] = self.h5.create_dataset('frame', shape=(0,), maxshape=(None,),
                                                            dtype='int64', chunks=True, compression='gzip', compression_opts=4)
            self.h5_dsets['timestamp'] = self.h5.create_dataset('timestamp', shape=(0,), maxshape=(None,),
                                                                dtype='float64', chunks=True, compression='gzip', compression_opts=4)
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
        for v in self.videos.values():
            try: v.close()
            except Exception: pass
        self.videos.clear()
        if self.csv:
            self.csv.flush(); self.csv.close(); self.csv = None
        import os as _os
        if self.h5 is not None:
            try:
                self.h5.flush(); self.h5.close()
                print(f"[record] episode.hdf5 saved at {_os.path.abspath(self.outdir)}/episode.hdf5")
            except Exception:
                pass
            self.h5 = None
        print(f"[record] STOP  -> {_os.path.abspath(self.outdir)}")
        self.enabled = False

    def step(self, chains, L_tgt, R_tgt):
        if not self.enabled or self.t0 is None:
            return
        import time, numpy as _np
        t_rel = time.time() - self.t0
        if t_rel - self.last + 1e-9 < self.interval:
            return
        self.last += self.interval
        frames_cache = {}
        for cam, rend in self.renderers.items():
            try:
                rend.update_scene(self.data, camera=cam)
                rgb = rend.render()
                frames_cache[cam] = rgb
                self.videos[cam].write(rgb)
            except Exception as e:
                print(f"[record] camera {cam}: {e}")
        # append to HDF5
        if self.h5 is not None and self.have_h5py:
            cam_map = {'top':'cam_high','left_wrist':'cam_left_wrist','right_wrist':'cam_right_wrist'}
            for cam, rgb in frames_cache.items():
                name = cam_map.get(cam, cam)
                ds = self.h5_dsets.get(name)
                if ds is not None:
                    ds.resize((ds.shape[0]+1,)+ds.shape[1:]); ds[-1, ...] = rgb
            def _append(name, arr):
                ds = self.h5_dsets.get(name)
                if ds is None: return
                ds.resize((ds.shape[0]+1,)+ds.shape[1:]); ds[-1, ...] = arr
            _append('qpos', _np.array(self.data.qpos[:], dtype='float64'))
            _append('qvel', _np.array(self.data.qvel[:], dtype='float64'))
            _append('L_tgt', _np.array(L_tgt, dtype='float32'))
            _append('R_tgt', _np.array(R_tgt, dtype='float32'))
            _append('frame', _np.array(self.frame_id, dtype='int64'))
            _append('timestamp', _np.array(t_rel, dtype='float64'))
        # csv
        Lp = self.data.xpos[chains["left"]["ee"]].copy();  Lq = self.data.xquat[chains["left"]["ee"]].copy()
        Rp = self.data.xpos[chains["right"]["ee"]].copy(); Rq = self.data.xquat[chains["right"]["ee"]].copy()
        row = [t_rel, self.frame_id] + [float(x) for x in self.data.qpos[:]] + \
              [*Lp.tolist(), *Lq.tolist(), *Rp.tolist(), *Rq.tolist(),
               float(L_tgt[0]), float(L_tgt[1]), float(L_tgt[2]),
               float(R_tgt[0]), float(R_tgt[1]), float(R_tgt[2])]
        try:
            self.writer.writerow(row)
        except Exception:
            pass
        self.frame_id += 1
# =============================================================


# ================== Live camera preview (OpenCV) ==================
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
        # Lazy import cv2
        try:
            import cv2 as _cv2
            self._cv2 = _cv2
        except Exception as e:
            print("[preview] OpenCV not available:", e, "-> run: pip install opencv-python")
        # Prepare renderers
        for name in self.cams:
            try:
                self.renderers[name] = mujoco.Renderer(model, self.size[0], self.size[1])
                print(f"[preview] renderer[{name}] size = {self.size[0]}x{self.size[1]}")
            except Exception as e:
                # Try swapped order
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
                rgb = rend.render()  # HxWx3 uint8
                # Convert to BGR for imshow
                bgr = rgb[..., ::-1].copy()
                # Draw cam name
                self._cv2.putText(bgr, cam, (8, 20), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, self._cv2.LINE_AA)
                frames.append(bgr)
            except Exception as e:
                print(f"[preview] {cam}: {e}")
        if not frames:
            return
        # Horizontal concat (pad to equal heights if necessary)
        try:
            panel = self._cv2.hconcat(frames)
        except Exception:
            # Fallback: make same width/height
            H = min(f.shape[0] for f in frames); W = min(f.shape[1] for f in frames)
            frames = [f[:H, :W] for f in frames]
            panel = self._cv2.hconcat(frames)
        self._cv2.imshow(self.window, panel)
        # Non-blocking wait
        self._cv2.waitKey(1)
# ================================================================

HELP = """
[Viewer+Terminal Controls] (keep TERMINAL focused)
  h    : help
  q    : quit
  0    : reset to keyframe 0
  [/]  : IK damping λ down/up
  -/=  : pos step down/up
  1/2/3: LEFT / RIGHT / BOTH
  g    : toggle go-to IK mode
  w/s, a/d, r/f : +Y/-Y, -X/+X, +Z/-Z (world)
  z/x  : left gripper close/open
  n/m  : right gripper close/open
  ,/.  : head_joint1 -/+
  ;/:  : head_joint2 -/+
  t/y  : platform_joint -/+
  R/S  : start/stop recording (3 cams)
  F    : toggle camera preview window (top | left_wrist | right_wrist)
        -> saves top.mp4, left_wrist.mp4, right_wrist.mp4, states.csv, meta.json, episode.hdf5
"""

def main():
    chains = find_arm_chain()
    print_chain(chains)

    mujoco.mj_forward(model, data)
    L_tgt = data.xpos[chains["left"]["ee"]].copy()
    R_tgt = data.xpos[chains["right"]["ee"]].copy()

    mode = "left"
    step = 0.01
    lam = 1e-3
    goto = True
    # Live camera preview (toggle with 'F')
    preview = CamPreview(model, data, cams=("top","left_wrist","right_wrist"), size=(320,240), fps=15)
    recorder = Recorder(model, data, cams=("top","left_wrist","right_wrist"), fps=30, size=(640,480))
    # joint debug watch (platform)
    platform_watch_enabled = False
    platform_watch_dt = 0.2  # 5 Hz
    platform_watch_last = 0.0

    # === Launch viewer EXACTLY like test6.py ===
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(3)  # test6.py does an initial pause

        # Delay terminal imports until after viewer is created
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
        with raw_terminal_mode(sys.stdin):
            while viewer.is_running():
                for ch in read_keys(0.0):
                    if ch in ('b','B'):
                        print(HELP)
                    elif ch == 'q':
                        return
                    elif ch == '0':
                        mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
                        L_tgt = data.xpos[chains["left"]["ee"]].copy()
                        R_tgt = data.xpos[chains["right"]["ee"]].copy()
                        print("[reset] keyframe 0")
                    elif ch == '[':
                        lam *= 1.5; print(f"[ik] λ -> {lam:.2e}")
                    elif ch == ']':
                        lam = max(1e-6, lam/1.5); print(f"[ik] λ -> {lam:.2e}")
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
                        # if mode in ('right','both'): R_tgt[1] += step
                    elif ch == 's':
                        if mode in ('left','both'): L_tgt[1] -= step
                        # if mode in ('right','both'): R_tgt[1] -= step
                    elif ch == 'a':
                        if mode in ('left','both'): L_tgt[0] -= step
                        # if mode in ('right','both'): R_tgt[0] -= step
                    elif ch == 'd':
                        if mode in ('left','both'): L_tgt[0] += step
                        # if mode in ('right','both'): R_tgt[0] += step
                    elif ch == 'r':
                        if mode in ('left','both'): L_tgt[2] += step
                        # if mode in ('right','both'): R_tgt[2] += step
                    elif ch == 'f':
                        if mode in ('left','both'): L_tgt[2] -= step
                        # if mode in ('right','both'): R_tgt[2] -= step
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
                    elif ch == 'z': set_gripper("left",  -0.005); print_gripper_state()
                    elif ch == 'x': set_gripper("left",  +0.005); print_gripper_state()
                    elif ch == 'n': set_gripper("right", -0.005); print_gripper_state()
                    elif ch == 'm': set_gripper("right", +0.005); print_gripper_state()
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

                if goto:
                    if mode in ('left','both'):  ik_step(chains, 'left',  L_tgt, lam=lam, dq_max=0.05)
                    if mode in ('right','both'): ik_step(chains, 'right', R_tgt, lam=lam, dq_max=0.05)

                # periodic platform watcher
                if platform_watch_enabled and (time.time() - platform_watch_last >= platform_watch_dt):
                    print_joint_debug('platform_joint')
                    platform_watch_last = time.time()

                recorder.step(chains, L_tgt, R_tgt)
                try:
                    import time as _t
                    preview.step(_t.perf_counter())
                except Exception:
                    pass
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
