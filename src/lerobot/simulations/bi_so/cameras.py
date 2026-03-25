#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

SIM_CAMERA_SPECS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "left_arm": {
        "model_name": "camera_left_arm",
        "aliases": ("left_arm",),
    },
    "right_arm": {
        "model_name": "camera_right_arm",
        "aliases": ("right_arm",),
    },
    "top": {
        "model_name": "camera_top",
        "aliases": ("top",),
    },
    "front": {
        "model_name": "camera_front",
        "aliases": ("front",),
    },
}

DEFAULT_RECORD_CAMERA_NAMES = ("left_arm", "right_arm", "top")


def default_camera_bridge_path() -> Path:
    return Path(__file__).resolve().with_name("bridge.py")


def default_camera_xml_path() -> Path:
    return Path(__file__).resolve().with_name("lerobot_pick_place_cube_cameras.xml")


def resolve_camera_names(
    camera_names: Sequence[str] | None,
    *,
    use_default_cameras: bool = False,
) -> tuple[str, ...]:
    requested_names = tuple(camera_names or ())
    if not requested_names and use_default_cameras:
        requested_names = DEFAULT_RECORD_CAMERA_NAMES

    unknown_names = sorted(set(requested_names) - set(SIM_CAMERA_SPECS))
    if unknown_names:
        supported = ", ".join(sorted(SIM_CAMERA_SPECS))
        raise ValueError(f"Unsupported camera names: {unknown_names}. Supported camera names: {supported}.")

    deduped_names: list[str] = []
    for camera_name in requested_names:
        if camera_name not in deduped_names:
            deduped_names.append(camera_name)
    return tuple(deduped_names)


def resolve_camera_assets(
    camera_names: Sequence[str],
    *,
    bridge_path: str | Path | None,
    xml_path: str | Path | None,
) -> tuple[Path | None, Path | None]:
    if not camera_names:
        resolved_bridge = None if bridge_path is None else Path(bridge_path)
        resolved_xml = None if xml_path is None else Path(xml_path)
        return resolved_bridge, resolved_xml

    resolved_bridge = default_camera_bridge_path() if bridge_path is None else Path(bridge_path)
    resolved_xml = default_camera_xml_path() if xml_path is None else Path(xml_path)
    return resolved_bridge, resolved_xml
