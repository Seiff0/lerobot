from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class Task2Sim:
    """Small MuJoCo wrapper for the local bimanual SO-arm scene."""

    def __init__(
        self,
        xml_path: str | Path,
        robot_dofs: int = 6,
        cube_raise_z: float = 0.05,
        substeps: int = 1,
        launch_viewer: bool = True,
        show_sites: bool = True,
        use_home_pose: bool = False,
        home_qpos: np.ndarray | None = None,
        home_ctrl: np.ndarray | None = None,
    ):
        self.xml_path = Path(xml_path).resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        self.robot_dofs = int(robot_dofs)
        self.substeps = int(substeps)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.ctrl_range = np.asarray(self.model.actuator_ctrlrange, dtype=float).copy()

        mujoco.mj_resetData(self.model, self.data)

        self.num_arms = max(1, int(self.model.nu) // self.robot_dofs)
        self.active_arm = 0
        self.viewer = None

        if use_home_pose and home_qpos is not None:
            self.apply_home_pose(home_qpos, home_ctrl)

        self._raise_cube_if_possible(float(cube_raise_z))
        mujoco.mj_forward(self.model, self.data)

        if launch_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self._key_callback,
            )
            if show_sites:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SITE] = 1
                except Exception:
                    pass

    def _key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except Exception:
            return

        if key == "t" and self.num_arms > 0:
            self.active_arm = (self.active_arm + 1) % self.num_arms
            print(f"[keyboard] active_arm={self.active_arm + 1}/{self.num_arms}")

    def _raise_cube_if_possible(self, z: float) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if body_id < 0:
            return

        joint_count = int(self.model.body_jntnum[body_id])
        if joint_count <= 0:
            return

        joint_id = int(self.model.body_jntadr[body_id])
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return

        qadr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qadr : qadr + 3] = np.array([0.2, 0.2, z], dtype=float)
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def apply_home_pose(self, home_qpos: np.ndarray, home_ctrl: np.ndarray | None = None) -> None:
        qpos = np.asarray(home_qpos, dtype=float).reshape(-1)
        ctrl = qpos if home_ctrl is None else np.asarray(home_ctrl, dtype=float).reshape(-1)

        joint_count = min(self.robot_dofs, qpos.size)
        for arm_index in range(self.num_arms):
            for joint_offset in range(joint_count):
                actuator_index = arm_index * self.robot_dofs + joint_offset
                if actuator_index >= int(self.model.nu):
                    break

                joint_id = int(self.model.actuator_trnid[actuator_index, 0])
                qadr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qadr] = qpos[joint_offset]

                dofadr = int(self.model.jnt_dofadr[joint_id])
                if 0 <= dofadr < int(self.model.nv):
                    self.data.qvel[dofadr] = 0.0

        if int(self.model.nu) > 0:
            ctrl_target = self.data.ctrl.copy()
            for arm_index in range(self.num_arms):
                start = arm_index * self.robot_dofs
                end = min(start + min(self.robot_dofs, ctrl.size), int(self.model.nu))
                if start >= end:
                    continue
                ctrl_target[start:end] = np.clip(
                    ctrl[: end - start],
                    self.ctrl_range[start:end, 0],
                    self.ctrl_range[start:end, 1],
                )
            self.data.ctrl[:] = ctrl_target

        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


@dataclass(frozen=True)
class _HomePose:
    qpos: np.ndarray
    ctrl: np.ndarray


@dataclass
class _SharedState:
    qpos_deg: np.ndarray
    images: dict[str, np.ndarray]


class Task2SharedBackend:
    """Shared MuJoCo backend for both arm bus views."""

    def __init__(
        self,
        xml_path: str,
        robot_dofs: int = 6,
        render_size: tuple[int, int] | None = (480, 640),
        realtime: bool = True,
        slowmo: float = 1.0,
        launch_viewer: bool = False,
    ):
        self.xml_path = Path(xml_path).resolve()
        self.sim = Task2Sim(
            xml_path=self.xml_path,
            robot_dofs=robot_dofs,
            launch_viewer=launch_viewer,
            show_sites=True,
        )
        self.model = self.sim.model
        self.data = self.sim.data

        self.robot_dofs = int(robot_dofs)
        self.nu = int(self.model.nu)
        self.num_arms = max(1, self.nu // self.robot_dofs)

        self.realtime = bool(realtime)
        self.slowmo = float(slowmo)

        self._ctrl_target = np.zeros(self.nu, dtype=float)

        if render_size is None:
            self._renderer = None
        else:
            height, width = render_size
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._refcount = 0
        self._state = _SharedState(
            qpos_deg=np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32),
            images={},
        )

    def _home_key_id(self) -> int:
        if int(self.model.nkey) <= 0:
            return -1
        try:
            return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        except Exception:
            return -1

    def _read_actuated_joint_qpos_rad(self) -> np.ndarray:
        qpos_rad = np.zeros(self.nu, dtype=float)
        for actuator_index in range(self.nu):
            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            qpos_rad[actuator_index] = float(self.data.qpos[qadr])
        return qpos_rad

    def _extract_first_arm_home_pose(self) -> _HomePose | None:
        key_id = self._home_key_id()
        if key_id < 0:
            return None

        model_key_qpos = np.asarray(self.model.key_qpos[key_id], dtype=float)
        model_key_ctrl = np.asarray(self.model.key_ctrl[key_id], dtype=float)

        qpos = np.zeros(self.robot_dofs, dtype=float)
        ctrl = np.zeros(self.robot_dofs, dtype=float)

        for joint_offset in range(self.robot_dofs):
            actuator_index = joint_offset
            if actuator_index >= self.nu:
                break

            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            if qadr >= model_key_qpos.size:
                return None

            qpos[joint_offset] = float(model_key_qpos[qadr])
            if actuator_index < model_key_ctrl.size:
                ctrl[joint_offset] = float(model_key_ctrl[actuator_index])
            else:
                ctrl[joint_offset] = qpos[joint_offset]

        return _HomePose(qpos=qpos, ctrl=ctrl)

    def _apply_startup_pose(self) -> None:
        home_pose = self._extract_first_arm_home_pose()
        if home_pose is not None:
            self.sim.apply_home_pose(home_pose.qpos, home_pose.ctrl)

        self._ctrl_target[:] = np.asarray(self.data.ctrl, dtype=float)
        if self.nu > 0:
            lo = self.model.actuator_ctrlrange[:, 0]
            hi = self.model.actuator_ctrlrange[:, 1]
            self._ctrl_target[:] = np.clip(self._ctrl_target, lo, hi)
            self.data.ctrl[:] = self._ctrl_target

        mujoco.mj_forward(self.model, self.data)

    def start(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._running:
                return
            # Apply the authored startup pose once, when the first consumer connects.
            if not np.any(self._ctrl_target) and np.allclose(self._read_actuated_joint_qpos_rad(), 0.0):
                self._apply_startup_pose()
            self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._refcount = max(0, self._refcount - 1)
            if self._refcount > 0:
                return
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        self.sim.close()

    def set_arm_target_deg(self, arm_index: int, q_deg: np.ndarray) -> None:
        q_deg = np.asarray(q_deg, dtype=float).reshape(-1)
        if q_deg.size != self.robot_dofs:
            raise ValueError(f"Expected {self.robot_dofs} values, got {q_deg.size}")

        start = arm_index * self.robot_dofs
        end = start + self.robot_dofs
        if end > self.nu:
            raise ValueError(f"arm_index {arm_index} out of range for nu={self.nu}")

        q_rad = np.deg2rad(q_deg)
        with self._lock:
            lo = self.model.actuator_ctrlrange[start:end, 0]
            hi = self.model.actuator_ctrlrange[start:end, 1]
            self._ctrl_target[start:end] = np.clip(q_rad, lo, hi)

    def get_state(self) -> _SharedState:
        with self._lock:
            return _SharedState(
                qpos_deg=self._state.qpos_deg.copy(),
                images={name: image.copy() for name, image in self._state.images.items()},
            )

    def _render_images(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            return {}

        images: dict[str, np.ndarray] = {}
        for camera_name in ("camera_front", "camera_top", "front", "top"):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                continue
            self._renderer.update_scene(self.data, camera=camera_name)
            images[camera_name] = self._renderer.render().copy()
        return images

    def _loop(self) -> None:
        dt = float(self.model.opt.timestep) * int(self.sim.substeps)

        while True:
            with self._lock:
                if not self._running:
                    break
                if self.nu > 0:
                    self.data.ctrl[:] = self._ctrl_target

            tick_start = time.time()
            self.sim.step()

            with self._lock:
                self._state.qpos_deg = np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32)
                self._state.images = self._render_images()

            if self.realtime:
                elapsed = time.time() - tick_start
                time.sleep(max(0.0, dt * self.slowmo - elapsed))


class Task2ArmBus:
    """Per-arm read/write wrapper matching the expected bus interface."""

    def __init__(self, backend: Task2SharedBackend, arm_index: int):
        self.backend = backend
        self.arm_index = int(arm_index)
        self.robot_dofs = backend.robot_dofs

        base = "joint_"
        suffix = "" if arm_index == 0 else "_r"
        self._motor_names = [f"{base}{i}{suffix}" for i in range(1, self.robot_dofs + 1)]

    @property
    def motor_names(self) -> list[str]:
        return self._motor_names

    def connect(self) -> None:
        self.backend.start()

    def disconnect(self) -> None:
        self.backend.stop()

    def read(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        state = self.backend.get_state()
        start = self.arm_index * self.robot_dofs
        end = start + self.robot_dofs
        return state.qpos_deg[start:end], state.images

    def write(self, *args, **kwargs) -> None:
        del kwargs
        if len(args) == 1:
            values = args[0]
        elif len(args) == 2:
            values = args[1]
        else:
            raise TypeError("write expects write(values) or write(group_name, values)")

        self.backend.set_arm_target_deg(self.arm_index, np.asarray(values, dtype=float))


def make_bimanual_buses(
    xml_path: str,
    robot_dofs: int = 6,
    render_size: tuple[int, int] | None = (480, 640),
    realtime: bool = True,
    slowmo: float = 1.0,
    launch_viewer: bool = False,
) -> tuple[Task2SharedBackend, dict[str, Task2ArmBus]]:
    backend = Task2SharedBackend(
        xml_path=xml_path,
        robot_dofs=robot_dofs,
        render_size=render_size,
        realtime=realtime,
        slowmo=slowmo,
        launch_viewer=launch_viewer,
    )
    buses = {f"arm{i}": Task2ArmBus(backend, i) for i in range(backend.num_arms)}
    return backend, buses


def make_task2_bimanual_buses(*args, **kwargs):
    """Compatibility alias for older configs/scripts."""
    return make_bimanual_buses(*args, **kwargs)
