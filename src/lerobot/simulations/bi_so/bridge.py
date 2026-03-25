#!/usr/bin/env python

from __future__ import annotations

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
        realtime: bool = True,
        slowmo: float = 1.0,
        launch_viewer: bool = False,
    ):
        super().__init__(
            xml_path=xml_path,
            robot_dofs=robot_dofs,
            render_size=render_size,
            realtime=realtime,
            slowmo=slowmo,
            launch_viewer=launch_viewer,
        )
        self._last_step_t: float | None = None

    def start(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._running:
                return
            if not np.any(self._ctrl_target) and np.allclose(self._read_actuated_joint_qpos_rad(), 0.0):
                self._apply_startup_pose()
            self._running = True
            self._last_step_t = time.perf_counter()
            self._state.qpos_deg = np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32)
            self._state.images = self._render_images()

    def stop(self) -> None:
        with self._lock:
            self._refcount = max(0, self._refcount - 1)
            if self._refcount > 0:
                return
            self._running = False

        self.sim.close()

    def _advance_simulation(self) -> None:
        if not self._running:
            return

        dt = float(self.model.opt.timestep) * int(self.sim.substeps)
        now = time.perf_counter()

        if self._last_step_t is None:
            step_count = 1
        elif self.realtime:
            target_elapsed = max(now - self._last_step_t, 0.0)
            step_count = max(1, int(round(target_elapsed / max(dt * self.slowmo, 1e-6))))
        else:
            step_count = 1

        if self.nu > 0:
            self.data.ctrl[:] = self._ctrl_target

        for _ in range(step_count):
            self.sim.step()

        self._last_step_t = now
        self._state.qpos_deg = np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32)
        self._state.images = self._render_images()

    def get_state(self):
        with self._lock:
            self._advance_simulation()
            return _SharedState(
                qpos_deg=self._state.qpos_deg.copy(),
                images={name: image.copy() for name, image in self._state.images.items()},
            )

    def _render_images(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            return {}

        images: dict[str, np.ndarray] = {}
        for spec in SIM_CAMERA_SPECS.values():
            model_name = str(spec["model_name"])
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, model_name)
            if camera_id < 0:
                continue

            self._renderer.update_scene(self.data, camera=model_name)
            rendered = self._renderer.render().copy()
            images[model_name] = rendered

            for alias in spec["aliases"]:
                images[str(alias)] = rendered

        return images


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
    return make_bimanual_buses(*args, **kwargs)
