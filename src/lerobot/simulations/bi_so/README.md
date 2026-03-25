# BI-SO MuJoCo Simulation

This simulation code is split into two folders on purpose:

- `src/lerobot/robots/bi_so_follower_simulated/`
  - the reusable robot package
  - this is where the MuJoCo bridge, XML scene, action mapping, and robot API live
- `src/lerobot/simulations/bi_so/`
  - the small user-facing runner scripts
  - this is where the commands for teleop, single-toggle, and viewer live

Keeping those separate makes the simulated robot usable from the normal LeRobot robot registry while still giving us dedicated scripts without modifying the main CLI files.

## Current File Layout

### Robot package

`src/lerobot/robots/bi_so_follower_simulated/config.py`
- config for the simulated robot
- selects the scene root, bridge file, XML file, viewer mode, cameras, and timing

`src/lerobot/robots/bi_so_follower_simulated/robot.py`
- main `BiSOFollowerSimulated` robot implementation
- loads the local MuJoCo backend
- exposes left/right action keys
- converts gripper values between SO teleop units and MuJoCo actuator units
- applies the joint offsets/sign fixes needed for this scene

`src/lerobot/robots/bi_so_follower_simulated/mujoco/bridge.py`
- combined MuJoCo backend file
- this replaces the older split between `task2_motors_bridge.py` and `mujoco_task2.py`
- contains:
  - scene loading
  - viewer handling
  - startup pose application
  - shared backend stepping loop
  - per-arm bus wrappers

`src/lerobot/robots/bi_so_follower_simulated/mujoco/lerobot_pick_place_cube.xml`
- the bimanual scene

`src/lerobot/robots/bi_so_follower_simulated/mujoco/so_arm100.xml`
- left arm model and authored `home` pose

`src/lerobot/robots/bi_so_follower_simulated/mujoco/so_arm100_right.xml`
- right arm model

`src/lerobot/robots/bi_so_follower_simulated/mujoco/assets/`
- STL meshes used by the MuJoCo scene

### Simulation scripts

`src/lerobot/simulations/bi_so/teleop.py`
- normal bimanual teleop entrypoint
- uses the standard `bi_so_leader` teleoperator path

`src/lerobot/simulations/bi_so/single_toggle.py`
- one real SO leader controls whichever simulated arm is active
- press `t` in the viewer to toggle between the two simulated arms

`src/lerobot/simulations/bi_so/view.py`
- launches the MuJoCo scene without connecting any real hardware

## Why We Did Not Collapse Everything Into One Folder

If everything lived only under `src/lerobot/simulations/bi_so/`, the simulated robot would stop being a normal LeRobot robot package.

The current split gives us both:
- a proper robot implementation under `robots/`
- small entrypoint scripts under `simulations/`

That is the cleanest way to avoid editing the main robot factory and teleop CLI files.

## Startup Behavior

The startup pose now comes from the authored MuJoCo `home` key in the arm XML and is applied to both arms by the local bridge when the first bus connects.

Important detail:
- the bridge holds that pose explicitly
- this avoids the older issue where the arm could drift away from the intended folded startup pose

## Control Paths

### 1. Standard bimanual teleop

Use this when you have two real leader arms and want the normal LeRobot teleoperation path:

```powershell
python -m lerobot.simulations.bi_so.teleop ^
  --robot.type=bi_so_follower_simulated ^
  --robot.sim_root=C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco ^
  --robot.launch_viewer=true ^
  --teleop.type=bi_so_leader ^
  --teleop.left_arm_config.port=COM5 ^
  --teleop.right_arm_config.port=COM6 ^
  --teleop.id=bimanual_leader ^
  --fps=60
```

### 2. Single leader with `t` toggle

Use this when one real leader arm should control one simulated arm at a time:

```powershell
python -m lerobot.simulations.bi_so.single_toggle ^
  --leader-port COM5 ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco ^
  --launch-viewer ^
  --hz 60
```

Behavior:
- viewer starts
- `t` switches the active simulated arm
- the single real leader drives the active arm only

### 3. Viewer only

Use this to inspect the scene without connecting any arms:

```powershell
python -m lerobot.simulations.bi_so.view ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco
```

## Mapping Notes

The simulated robot exposes:
- `left_shoulder_pan.pos`
- `left_shoulder_lift.pos`
- `left_elbow_flex.pos`
- `left_wrist_flex.pos`
- `left_wrist_roll.pos`
- `left_gripper.pos`
- and the same keys with the `right_` prefix

Internally:
- the first five joints are treated as angle values
- the gripper is exposed as SO-style `0..100`
- the robot layer converts that gripper value to the MuJoCo actuator range
- the robot layer also applies the static joint offset and sign fix required by this scene

## What Changed Over Time

This simulation started with copied files and longer names. It has now been cleaned into:

- `config.py`
- `robot.py`
- `mujoco/bridge.py`
- `teleop.py`
- `single_toggle.py`
- `view.py`

The old split `task2_motors_bridge.py` + `mujoco_task2.py` was combined into one clearer backend file:

- `src/lerobot/robots/bi_so_follower_simulated/mujoco/bridge.py`

The old longer entrypoint names were shortened to match what they actually do:

- `teleop.py`
- `single_toggle.py`
- `view.py`

## Known Limitations

- The simulation scripts are intentionally local and isolated; they do not modify the main LeRobot CLI registration flow.
- The single-toggle mode is a custom script, not part of the standard `bi_so_leader` teleoperator.
- The MuJoCo scene still uses hand-tuned mapping offsets in the robot wrapper.

## Fast Debugging Guide

If the problem is:

- startup pose or viewer behavior
  - look in `mujoco/bridge.py`
- robot action mapping or gripper conversion
  - look in `robot.py`
- robot config and path resolution
  - look in `config.py`
- one-controller toggle behavior
  - look in `single_toggle.py`
- normal two-leader teleop launch
  - look in `teleop.py`
- viewer-only launch
  - look in `view.py`
