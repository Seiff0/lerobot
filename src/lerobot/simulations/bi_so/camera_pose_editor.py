#!/usr/bin/env python

from __future__ import annotations

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.utils.utils import init_logging


@dataclass(frozen=True)
class CameraEntry:
    name: str
    file_path: Path
    camera_id: int
    parent_body_name: str | None
    proxy_body_name: str


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def _matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=float)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    return _normalize(np.array([qw, qx, qy, qz], dtype=float))


def _camera_world_pose(model: mujoco.MjModel, data: mujoco.MjData, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(data.cam_xpos[camera_id], dtype=float).copy()
    rot = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3).copy()
    return pos, rot


def _body_world_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str | None) -> tuple[np.ndarray, np.ndarray]:
    if body_name is None:
        return np.zeros(3, dtype=float), np.eye(3, dtype=float)

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"Unknown body '{body_name}' in temporary model.")

    pos = np.asarray(data.xpos[body_id], dtype=float).copy()
    rot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return pos, rot


def _viewer_world_pose(viewer) -> tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(float(viewer.cam.azimuth))
    elevation_rad = np.deg2rad(float(viewer.cam.elevation))
    lookat = np.asarray(viewer.cam.lookat, dtype=float)
    distance = float(viewer.cam.distance)

    forward = np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=float,
    )
    forward = _normalize(forward)
    pos = lookat - (distance * forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)

    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    rot = np.column_stack((right, up, -forward))
    return pos, rot


def _proxy_name(camera_name: str) -> str:
    return f"camera_pose_proxy__{camera_name}"


def _collect_camera_entries(
    root_xml_path: Path,
    model: mujoco.MjModel,
    xml_path: Path | None = None,
    visited: set[Path] | None = None,
    entries: dict[str, CameraEntry] | None = None,
) -> dict[str, CameraEntry]:
    xml_path = root_xml_path if xml_path is None else xml_path
    visited = set() if visited is None else visited
    entries = {} if entries is None else entries

    xml_path = xml_path.resolve()
    if xml_path in visited:
        return entries
    visited.add(xml_path)

    tree = ET.parse(xml_path)
    xml_root = tree.getroot()

    for child in xml_root:
        if child.tag != "include":
            continue
        include_file = child.get("file")
        if not include_file:
            continue
        include_path = (xml_path.parent / include_file).resolve()
        _collect_camera_entries(root_xml_path, model, include_path, visited, entries)

    body_name_by_elem_id: dict[int, str | None] = {id(xml_root): None}
    for elem in xml_root.iter():
        if elem.tag == "body":
            body_name_by_elem_id[id(elem)] = elem.get("name")
        for child in list(elem):
            body_name_by_elem_id[id(child)] = body_name_by_elem_id.get(id(elem))

    for elem in xml_root.iter("camera"):
        camera_name = elem.get("name")
        if not camera_name:
            continue
        if camera_name in entries:
            raise ValueError(f"Duplicate camera name '{camera_name}' found while parsing XML files.")

        camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
        if camera_id < 0:
            continue

        entries[camera_name] = CameraEntry(
            name=camera_name,
            file_path=xml_path,
            camera_id=camera_id,
            parent_body_name=body_name_by_elem_id.get(id(elem)),
            proxy_body_name=_proxy_name(camera_name),
        )

    return entries


def _write_camera_pose(entry: CameraEntry, local_pos: np.ndarray, local_quat: np.ndarray) -> None:
    tree = ET.parse(entry.file_path)
    xml_root = tree.getroot()

    target_elem = None
    for elem in xml_root.iter("camera"):
        if elem.get("name") == entry.name:
            target_elem = elem
            break

    if target_elem is None:
        raise ValueError(f"Could not find camera '{entry.name}' in {entry.file_path}.")

    target_elem.set("pos", _fmt(local_pos))
    target_elem.set("quat", _fmt(local_quat))
    for attr in ("xyaxes", "euler", "axisangle", "zaxis"):
        if attr in target_elem.attrib:
            del target_elem.attrib[attr]

    ET.indent(tree, space="  ")
    tree.write(entry.file_path, encoding="utf-8", xml_declaration=False)


def _build_temp_xml(root_xml_path: Path, model: mujoco.MjModel, data: mujoco.MjData, entries: list[CameraEntry]) -> Path:
    tree = ET.parse(root_xml_path)
    xml_root = tree.getroot()
    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in {root_xml_path}.")

    for entry in entries:
        world_pos, world_rot = _camera_world_pose(model, data, entry.camera_id)
        body = ET.SubElement(
            worldbody,
            "body",
            name=entry.proxy_body_name,
            pos=_fmt(world_pos),
            quat=_fmt(_matrix_to_quat(world_rot)),
        )
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            type="sphere",
            size="0.03",
            rgba="1 0.85 0.15 0.95",
            contype="0",
            conaffinity="0",
            density="10",
        )

    ET.indent(tree, space="  ")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix="camera_pose_editor_",
        dir=root_xml_path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        tree.write(handle, encoding="unicode", xml_declaration=False)
        return Path(handle.name)


def _proxy_qpos_addr(model: mujoco.MjModel, body_name: str) -> int:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"Unknown proxy body '{body_name}'.")
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0:
        raise ValueError(f"Proxy body '{body_name}' has no joint.")
    return int(model.jnt_qposadr[joint_id])


def _set_proxy_world_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    world_pos: np.ndarray,
    world_rot: np.ndarray,
) -> None:
    qpos_adr = _proxy_qpos_addr(model, body_name)
    data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(world_pos, dtype=float)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = _matrix_to_quat(world_rot)
    mujoco.mj_forward(model, data)


def _proxy_world_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"Unknown proxy body '{body_name}'.")
    pos = np.asarray(data.xpos[body_id], dtype=float).copy()
    rot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return pos, rot


def _proxy_positions(model: mujoco.MjModel, data: mujoco.MjData, entries: list[CameraEntry]) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for entry in entries:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entry.proxy_body_name))
        if body_id >= 0:
            positions.append(np.asarray(data.xpos[body_id], dtype=float).copy())
    return positions


def _focus_on_entries(viewer, model: mujoco.MjModel, data: mujoco.MjData, entries: list[CameraEntry]) -> None:
    positions = _proxy_positions(model, data, entries)
    if not positions:
        return

    stacked = np.vstack(positions)
    center = stacked.mean(axis=0)
    spread = float(np.max(np.linalg.norm(stacked - center, axis=1))) if len(positions) > 1 else 0.0

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = center
    viewer.cam.distance = max(0.35, 0.25 + (2.5 * spread))
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -20.0
    viewer.sync()


def _focus_on_entry(viewer, model: mujoco.MjModel, data: mujoco.MjData, entry: CameraEntry) -> None:
    world_pos, _ = _proxy_world_pose(model, data, entry.proxy_body_name)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = world_pos
    viewer.cam.distance = 0.25
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -20.0
    viewer.sync()


def _apply_proxy_to_camera(model: mujoco.MjModel, data: mujoco.MjData, entry: CameraEntry) -> tuple[np.ndarray, np.ndarray]:
    proxy_pos, proxy_rot = _proxy_world_pose(model, data, entry.proxy_body_name)
    parent_pos, parent_rot = _body_world_pose(model, data, entry.parent_body_name)
    local_pos = parent_rot.T @ (proxy_pos - parent_pos)
    local_rot = parent_rot.T @ proxy_rot
    local_quat = _matrix_to_quat(local_rot)
    model.cam_pos[entry.camera_id] = local_pos
    model.cam_quat[entry.camera_id] = local_quat
    mujoco.mj_forward(model, data)
    return local_pos, local_quat


def _capture_viewer_pose(model: mujoco.MjModel, data: mujoco.MjData, entry: CameraEntry, viewer) -> tuple[np.ndarray, np.ndarray]:
    world_pos, world_rot = _viewer_world_pose(viewer)
    _set_proxy_world_pose(model, data, entry.proxy_body_name, world_pos, world_rot)
    return _apply_proxy_to_camera(model, data, entry)


def _nudge_proxy_local(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    entry: CameraEntry,
    delta_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_pos, world_rot = _proxy_world_pose(model, data, entry.proxy_body_name)
    world_delta = world_rot @ np.asarray(delta_local, dtype=float)
    _set_proxy_world_pose(model, data, entry.proxy_body_name, world_pos + world_delta, world_rot)
    return _apply_proxy_to_camera(model, data, entry)


def _print_camera_status(model: mujoco.MjModel, data: mujoco.MjData, entry: CameraEntry) -> None:
    world_pos, world_rot = _proxy_world_pose(model, data, entry.proxy_body_name)
    local_pos, local_quat = _apply_proxy_to_camera(model, data, entry)
    print()
    print(f"camera: {entry.name}")
    print(f"file:   {entry.file_path}")
    print(f"world pos: {_fmt(world_pos)}")
    print(f"world quat: {_fmt(_matrix_to_quat(world_rot))}")
    print(f"local pos: {_fmt(local_pos)}")
    print(f"local quat: {_fmt(local_quat)}")


def _save_entry(model: mujoco.MjModel, data: mujoco.MjData, entry: CameraEntry) -> None:
    local_pos, local_quat = _apply_proxy_to_camera(model, data, entry)
    _write_camera_pose(entry, local_pos, local_quat)
    print()
    print(f"saved {entry.name} to {entry.file_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit MuJoCo camera poses by moving the viewer with the mouse and capturing that pose into XML cameras."
    )
    parser.add_argument(
        "--xml-path",
        default=str(Path(__file__).resolve().with_name("lerobot_pick_place_cube_cameras.xml")),
        help="Top-level XML to load.",
    )
    parser.add_argument("--camera", default=None, help="Optional camera to select first.")
    parser.add_argument("--fps", type=float, default=60.0, help="Viewer refresh rate.")
    parser.add_argument("--move-step", type=float, default=0.01, help="Keyboard translation step in scene units.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    init_logging()

    root_xml_path = Path(args.xml_path).resolve()
    if not root_xml_path.exists():
        raise FileNotFoundError(f"XML not found: {root_xml_path}")

    base_model = mujoco.MjModel.from_xml_path(str(root_xml_path))
    base_data = mujoco.MjData(base_model)
    key_id = int(mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(base_model, base_data, key_id)
    else:
        mujoco.mj_resetData(base_model, base_data)
    mujoco.mj_forward(base_model, base_data)

    entries_by_name = _collect_camera_entries(root_xml_path, base_model)
    if not entries_by_name:
        raise ValueError(f"No cameras found in {root_xml_path}.")

    camera_names = sorted(entries_by_name)
    current_index = 0
    move_step = float(args.move_step)
    if args.camera is not None:
        if args.camera not in entries_by_name:
            supported = ", ".join(camera_names)
            raise ValueError(f"Unknown camera '{args.camera}'. Available cameras: {supported}")
        current_index = camera_names.index(args.camera)

    temp_xml_path = _build_temp_xml(root_xml_path, base_model, base_data, [entries_by_name[name] for name in camera_names])
    try:
        model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
        data = mujoco.MjData(model)
        key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        def current_entry() -> CameraEntry:
            return entries_by_name[camera_names[current_index]]

        def select_camera(new_index: int, viewer) -> None:
            nonlocal current_index
            current_index = new_index % len(camera_names)
            entry = current_entry()
            _focus_on_entry(viewer, model, data, entry)
            print()
            print(f"selected: {entry.name}")
            print(f"source:   {entry.file_path}")

        def view_selected_camera(viewer) -> None:
            entry = current_entry()
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = entry.camera_id
            viewer.sync()

        viewer_holder: dict[str, object] = {}

        def key_callback(keycode: int) -> None:
            viewer = viewer_holder.get("viewer")
            if viewer is None:
                return

            try:
                key = chr(keycode).lower()
            except Exception:
                return

            if key == "n":
                select_camera(current_index + 1, viewer)
            elif key == "b":
                select_camera(current_index - 1, viewer)
            elif key == "f":
                _focus_on_entry(viewer, model, data, current_entry())
            elif key == "g":
                _focus_on_entries(viewer, model, data, [entries_by_name[name] for name in camera_names])
            elif key == "v":
                view_selected_camera(viewer)
            elif key == "c":
                if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                    print("switch to free-view first with 'f' or 'g', move the mouse, then press 'c' to capture.")
                else:
                    _capture_viewer_pose(model, data, current_entry(), viewer)
                    _print_camera_status(model, data, current_entry())
            elif key == "j":
                _nudge_proxy_local(model, data, current_entry(), np.array([-move_step, 0.0, 0.0], dtype=float))
                _print_camera_status(model, data, current_entry())
            elif key == "l":
                _nudge_proxy_local(model, data, current_entry(), np.array([move_step, 0.0, 0.0], dtype=float))
                _print_camera_status(model, data, current_entry())
            elif key == "i":
                _nudge_proxy_local(model, data, current_entry(), np.array([0.0, 0.0, -move_step], dtype=float))
                _print_camera_status(model, data, current_entry())
            elif key == "k":
                _nudge_proxy_local(model, data, current_entry(), np.array([0.0, 0.0, move_step], dtype=float))
                _print_camera_status(model, data, current_entry())
            elif key == "u":
                _nudge_proxy_local(model, data, current_entry(), np.array([0.0, move_step, 0.0], dtype=float))
                _print_camera_status(model, data, current_entry())
            elif key == "o":
                _nudge_proxy_local(model, data, current_entry(), np.array([0.0, -move_step, 0.0], dtype=float))
                _print_camera_status(model, data, current_entry())
            elif key == "m":
                _save_entry(model, data, current_entry())
            elif key == "a":
                for name in camera_names:
                    _save_entry(model, data, entries_by_name[name])
            elif key == "p":
                _print_camera_status(model, data, current_entry())
            elif key == "h":
                print("keys: n/b select, f focus selected, g focus all, v view camera, c capture viewer, j/l left-right, i/k forward-back, u/o up-down, m save selected, a save all, p print pose")

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer_holder["viewer"] = viewer
            _focus_on_entries(viewer, model, data, [entries_by_name[name] for name in camera_names])
            print("camera pose editor")
            print("  n / b : next / previous camera")
            print("  f     : free-view focus on selected camera marker")
            print("  g     : free-view focus on all camera markers")
            print("  v     : look through the selected camera")
            print("  c     : capture the current free-view mouse pose into the selected camera")
            print("  j / l : move selected camera left / right")
            print("  i / k : move selected camera forward / backward")
            print("  u / o : move selected camera up / down")
            print("  m     : save the selected camera to XML")
            print("  a     : save all cameras to XML")
            print("  p     : print the selected camera pose")
            print("  h     : print the shortcuts again")
            print()
            print("Workflow: select a camera, press 'f', move the mouse until the view looks right, press 'c', then 'm'.")
            select_camera(current_index, viewer)

            while viewer.is_running():
                viewer.sync()
                time.sleep(max(1.0 / float(args.fps), 0.001))
    finally:
        try:
            temp_xml_path.unlink(missing_ok=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
