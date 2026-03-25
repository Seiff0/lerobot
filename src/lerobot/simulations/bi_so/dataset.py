#!/usr/bin/env python

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.bi_so_follower_simulated.config import BiSOFollowerSimulatedConfig
from lerobot.robots.bi_so_follower_simulated.robot import BiSOFollowerSimulated, MOTOR_NAMES
from lerobot.simulations.bi_so.single_toggle import _active_arm_index
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def _add_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sim-root", default=None)
    parser.add_argument("--bridge-path", default=None)
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--bridge-factory-name", default="make_bimanual_buses")
    parser.add_argument("--robot-dofs", type=int, default=6)
    parser.add_argument("--launch-viewer", action="store_true", default=False)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--camera-names", nargs="*", default=())
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=640)


def _build_sim_helper(args: argparse.Namespace, *, sim_id: str) -> BiSOFollowerSimulated:
    render_size = None if not args.camera_names else (args.render_height, args.render_width)
    cfg = BiSOFollowerSimulatedConfig(
        id=sim_id,
        sim_root=None if args.sim_root is None else Path(args.sim_root),
        bridge_path=None if args.bridge_path is None else Path(args.bridge_path),
        xml_path=None if args.xml_path is None else Path(args.xml_path),
        bridge_factory_name=args.bridge_factory_name,
        robot_dofs=args.robot_dofs,
        render_size=render_size,
        camera_names=tuple(args.camera_names),
        realtime=args.realtime,
        slowmo=args.slowmo,
        launch_viewer=args.launch_viewer,
    )
    return BiSOFollowerSimulated(cfg)


def _leader_action_to_robot_array(action: dict[str, float]) -> list[float]:
    return [float(action[f"{motor_name}.pos"]) for motor_name in MOTOR_NAMES]


def _get_dataset_features(robot: BiSOFollowerSimulated, args: argparse.Namespace) -> dict[str, dict]:
    return combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=args.video),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=args.video),
    )


def _get_dataset_root(args: argparse.Namespace) -> Path:
    return Path(args.root) if args.root is not None else HF_LEROBOT_HOME / args.repo_id


def _create_dataset(robot: BiSOFollowerSimulated, args: argparse.Namespace) -> LeRobotDataset:
    features = _get_dataset_features(robot, args)

    if args.resume:
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        sanity_check_dataset_robot_compatibility(dataset, robot, int(args.fps), features)
        return dataset

    try:
        return LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=int(args.fps),
            features=features,
            root=args.root,
            robot_type=robot.robot_type,
            use_videos=args.video,
        )
    except FileExistsError as exc:
        dataset_root = _get_dataset_root(args)
        raise FileExistsError(
            f"Dataset already exists at '{dataset_root}'. Use '--resume' to append episodes, "
            "or choose a new '--repo-id'/'--root'."
        ) from exc


def _current_action_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(observation[key]) for key in observation if key.endswith(".pos")}


def _load_episode_actions(repo_id: str, episode: int, root: str | Path | None = None) -> tuple[LeRobotDataset, Any]:
    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    actions = episode_frames.select_columns(ACTION)
    return dataset, actions


def _record(args: argparse.Namespace) -> int:
    sim_helper = _build_sim_helper(args, sim_id="bimanual_so_dataset_record")
    leader_cfg = SOLeaderTeleopConfig(
        id=args.leader_id,
        calibration_dir=None if args.leader_calibration_dir is None else Path(args.leader_calibration_dir),
        port=args.leader_port,
        use_degrees=args.leader_use_degrees,
    )
    leader = SOLeader(leader_cfg)
    dataset = None

    try:
        sim_helper.connect()
        leader.connect(calibrate=args.calibrate)
        dataset = _create_dataset(sim_helper, args)

        for _ in range(int(args.num_episodes)):
            if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                dataset.clear_episode_buffer()

            episode_start = time.perf_counter()
            while time.perf_counter() - episode_start < float(args.episode_time_s):
                tick_start = time.perf_counter()

                observation = sim_helper.get_observation()
                action = _current_action_from_observation(observation)

                active_arm = _active_arm_index(sim_helper._backend)
                arm_prefix = "left" if active_arm == 0 else "right"
                leader_action = leader.get_action()
                leader_values = _leader_action_to_robot_array(leader_action)
                for motor_name, value in zip(MOTOR_NAMES, leader_values, strict=True):
                    action[f"{arm_prefix}_{motor_name}.pos"] = float(value)

                sent_action = sim_helper.send_action(action)
                observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": args.task}
                dataset.add_frame(frame)

                dt_s = time.perf_counter() - tick_start
                precise_sleep(max((1.0 / float(args.fps)) - dt_s, 0.0))

            dataset.save_episode()

        dataset.finalize()
        return 0
    except KeyboardInterrupt:
        if dataset is not None:
            dataset.finalize()
        return 0
    finally:
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            sim_helper.disconnect()
        except Exception:
            pass


def _replay(args: argparse.Namespace) -> int:
    sim_helper = _build_sim_helper(args, sim_id="bimanual_so_dataset_replay")
    try:
        sim_helper.connect()
        dataset, actions = _load_episode_actions(repo_id=args.repo_id, episode=args.episode, root=args.root)
        action_names = dataset.features[ACTION]["names"]
        replay_fps = dataset.fps if args.fps is None else int(args.fps)

        for idx in range(len(actions)):
            start = time.perf_counter()
            action_values = actions[idx][ACTION]
            sim_helper.send_action({name: action_values[i] for i, name in enumerate(action_names)})
            dt_s = time.perf_counter() - start
            precise_sleep(max((1.0 / replay_fps) - dt_s, 0.0))
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            sim_helper.disconnect()
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--leader-port", required=True)
    record_parser.add_argument("--leader-id", default="single_so_leader")
    record_parser.add_argument("--leader-calibration-dir", default=None)
    record_parser.add_argument("--leader-use-degrees", action="store_true", default=True)
    record_parser.add_argument("--no-leader-use-degrees", dest="leader_use_degrees", action="store_false")
    record_parser.add_argument("--calibrate", action="store_true", default=True)
    record_parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    record_parser.add_argument("--repo-id", required=True)
    record_parser.add_argument("--root", default=None)
    record_parser.add_argument("--resume", action="store_true", default=False)
    record_parser.add_argument("--task", required=True)
    record_parser.add_argument("--fps", type=int, default=30)
    record_parser.add_argument("--episode-time-s", type=float, default=30.0)
    record_parser.add_argument("--num-episodes", type=int, default=1)
    record_parser.add_argument("--video", action="store_true", default=True)
    record_parser.add_argument("--no-video", dest="video", action="store_false")
    _add_sim_args(record_parser)
    record_parser.set_defaults(launch_viewer=True)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--repo-id", required=True)
    replay_parser.add_argument("--root", default=None)
    replay_parser.add_argument("--episode", type=int, default=0)
    replay_parser.add_argument("--fps", type=int, default=None)
    _add_sim_args(replay_parser)
    replay_parser.set_defaults(launch_viewer=True)

    return parser


def main() -> int:
    init_logging()
    args = _build_parser().parse_args()
    if args.mode == "record":
        return _record(args)
    if args.mode == "replay":
        return _replay(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
