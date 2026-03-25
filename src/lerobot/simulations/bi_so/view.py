#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open the bimanual MuJoCo scene without connecting any real hardware."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot.robots.bi_so_follower_simulated import BiSOFollowerSimulated, BiSOFollowerSimulatedConfig
from lerobot.utils.utils import init_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", default=None)
    parser.add_argument("--bridge-path", default=None)
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--bridge-factory-name", default="make_bimanual_buses")
    parser.add_argument("--robot-dofs", type=int, default=6)
    parser.add_argument("--launch-viewer", action="store_true", default=True)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.set_defaults(launch_viewer=True)
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def _build_sim_helper(args: argparse.Namespace) -> BiSOFollowerSimulated:
    cfg = BiSOFollowerSimulatedConfig(
        id="bimanual_so_follower_viewer",
        sim_root=None if args.sim_root is None else Path(args.sim_root),
        bridge_path=None if args.bridge_path is None else Path(args.bridge_path),
        xml_path=None if args.xml_path is None else Path(args.xml_path),
        bridge_factory_name=args.bridge_factory_name,
        robot_dofs=args.robot_dofs,
        realtime=args.realtime,
        slowmo=args.slowmo,
        launch_viewer=args.launch_viewer,
    )
    return BiSOFollowerSimulated(cfg)


def main() -> int:
    args = _parse_args()
    init_logging()

    sim_helper = _build_sim_helper(args)
    try:
        sim_helper.connect()
        while True:
            time.sleep(max(1.0 / float(args.fps), 0.001))
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            sim_helper.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
