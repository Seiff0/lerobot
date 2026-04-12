#!/usr/bin/env python

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np

from lerobot.robots.bi_so_follower_simulated.mujoco.bridge import (
    Task2ArmBus,
    Task2SharedBackend as _BaseTask2SharedBackend,
    _SharedState,
)
from lerobot.simulations.bi_so.cameras import SIM_CAMERA_SPECS


class Task2SharedBackend(_BaseTask2SharedBackend):
    """Simulation-local bridge that renders the extra dataset cameras."""

    def __init__(
        self,
        xml_path: str,
        robot_dofs: int = 6,
        render_size: tuple[int, int] | None = (480, 640),
        camera_names: tuple[str, ...] = (),
        camera_fps: float | None = None,
        realtime: bool = True,
        slowmo: float = 1.0,
        launch_viewer: bool = False,
    ):
        super().__init__(
            xml_path=xml_path,
            robot_dofs=robot_dofs,
            render_size=render_size,
            camera_names=camera_names,
            camera_fps=camera_fps,
            realtime=realtime,
            slowmo=slowmo,
            launch_viewer=launch_viewer,
        )
        requested_names = tuple(camera_names or SIM_CAMERA_SPECS.keys())
        self._camera_specs = [SIM_CAMERA_SPECS[name] for name in requested_names if name in SIM_CAMERA_SPECS]
        self._camera_period_s = None if camera_fps is None or camera_fps <= 0 else 1.0 / float(camera_fps)
        self._last_render_t: float | None = None

    def _ensure_renderer(self) -> None:
        if self.render_size is None or self._renderer is not None:
            return
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    def start(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._running:
                return
            if not np.any(self._ctrl_target) and np.allclose(self._read_actuated_joint_qpos_rad(), 0.0):
                self._apply_startup_pose()
            self._ensure_renderer()
            self._running = True
            self._state.qpos_deg = np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32)
            self._state.images = {}

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

    def _loop(self) -> None:
        dt = float(self.model.opt.timestep) * int(self.sim.substeps)

        while True:
            with self._lock:
                if not self._running:
                    break
                if self.nu > 0:
                    self.data.ctrl[:] = self._ctrl_target

            tick_start = time.perf_counter()
            self.sim.step()

            with self._lock:
                self._state.qpos_deg = np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32)

            if self.realtime:
                elapsed = time.perf_counter() - tick_start
                time.sleep(max(0.0, dt * self.slowmo - elapsed))

    def get_state(self):
        with self._lock:
            self._state.images = self._render_images()
            return _SharedState(
                qpos_deg=self._state.qpos_deg.copy(),
                images={name: image.copy() for name, image in self._state.images.items()},
            )

    def peek_state(self):
        with self._lock:
            return _SharedState(
                qpos_deg=self._state.qpos_deg.copy(),
                images={name: image.copy() for name, image in self._state.images.items()},
            )

    def _render_images(self) -> dict[str, np.ndarray]:
        self._ensure_renderer()
        if self._renderer is None:
            return {}

        now = time.perf_counter()
        if (
            self._camera_period_s is not None
            and self._last_render_t is not None
            and (now - self._last_render_t) < self._camera_period_s
            and self._state.images
        ):
            return {name: image.copy() for name, image in self._state.images.items()}

        images: dict[str, np.ndarray] = {}
        for spec in self._camera_specs:
            model_name = str(spec["model_name"])
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, model_name)
            if camera_id < 0:
                continue

            self._renderer.update_scene(self.data, camera=model_name)
            rendered = self._renderer.render().copy()
            images[model_name] = rendered

            for alias in spec["aliases"]:
                images[str(alias)] = rendered

        self._last_render_t = now
        return images


def make_bimanual_buses(
    xml_path: str,
    robot_dofs: int = 6,
    render_size: tuple[int, int] | None = (480, 640),
    camera_names: tuple[str, ...] = (),
    camera_fps: float | None = None,
    realtime: bool = True,
    slowmo: float = 1.0,
    launch_viewer: bool = False,
) -> tuple[Task2SharedBackend, dict[str, Task2ArmBus]]:
    backend = Task2SharedBackend(
        xml_path=xml_path,
        robot_dofs=robot_dofs,
        render_size=render_size,
        camera_names=camera_names,
        camera_fps=camera_fps,
        realtime=realtime,
        slowmo=slowmo,
        launch_viewer=launch_viewer,
    )
    buses = {f"arm{i}": Task2ArmBus(backend, i) for i in range(backend.num_arms)}
    return backend, buses


def make_task2_bimanual_buses(*args, **kwargs):
    return make_bimanual_buses(*args, **kwargs)
