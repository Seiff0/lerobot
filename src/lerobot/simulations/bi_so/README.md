# BI-SO MuJoCo Simulation

This simulation is split into two places on purpose:

- `src/lerobot/robots/bi_so_follower_simulated/`
  This is the reusable robot package. It contains the MuJoCo bridge, XML scene, robot config, and action mapping.
- `src/lerobot/simulations/bi_so/`
  This is the user-facing script folder. It contains the small commands used to run the simulation.

That keeps the simulated robot usable from the normal LeRobot robot registry while avoiding changes to the main CLI files.

## Current Files

### Robot package

`src/lerobot/robots/bi_so_follower_simulated/config.py`
- config for the simulated robot

`src/lerobot/robots/bi_so_follower_simulated/robot.py`
- `BiSOFollowerSimulated`
- loads the local MuJoCo backend
- exposes left/right action keys
- converts gripper values between SO teleop units and MuJoCo actuator units
- applies the scene-specific joint offset/sign mapping

`src/lerobot/robots/bi_so_follower_simulated/mujoco/bridge.py`
- combined MuJoCo backend file
- contains scene loading, viewer handling, startup pose logic, stepping, and per-arm bus wrappers

`src/lerobot/robots/bi_so_follower_simulated/mujoco/lerobot_pick_place_cube.xml`
- the bimanual scene

`src/lerobot/robots/bi_so_follower_simulated/mujoco/so_arm100.xml`
- left arm model and authored `home` pose

`src/lerobot/robots/bi_so_follower_simulated/mujoco/so_arm100_right.xml`
- right arm model

### Simulation scripts

`src/lerobot/simulations/bi_so/teleop.py`
- normal bimanual teleop entrypoint

`src/lerobot/simulations/bi_so/single_toggle.py`
- one real SO leader controls whichever simulated arm is active
- press `t` in the viewer to switch the active arm

`src/lerobot/simulations/bi_so/view.py`
- launches the scene without any real hardware

`src/lerobot/simulations/bi_so/dataset.py`
- AOSH-style dataset entrypoint
- provides `record` and `replay` subcommands in one file
- `record` supports either one SO leader arm (`--leader-port`) or a bimanual pair (`--left-leader-port` and `--right-leader-port`)

`src/lerobot/simulations/bi_so/cameras.py`
- simulation-local camera presets and default camera asset paths

`src/lerobot/simulations/bi_so/bridge.py`
- simulation-local MuJoCo bridge that renders left-arm, right-arm, top, and front cameras

`src/lerobot/simulations/bi_so/lerobot_pick_place_cube_cameras.xml`
- simulation-local scene file with extra recording cameras

`src/lerobot/simulations/bi_so/camera_pose_editor.py`
- interactive editor for the simulated camera XML poses

## Commands

### 1. Standard bimanual teleop

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

```powershell
python -m lerobot.simulations.bi_so.single_toggle ^
  --leader-port COM5 ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco ^
  --launch-viewer ^
  --hz 60
```

### 3. Viewer only

```powershell
python -m lerobot.simulations.bi_so.view ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco
```

### 4. Record a dataset

This follows the AOSH-style simulation flow: one script, `record` mode, dataset arguments like `--repo-id`, `--root`, `--episode-time-s`, and `--num-episodes`.

Single-controller mode:
- use `--leader-port COM5`
- the controller drives whichever simulated arm is currently active

Dual-controller mode:
- use both `--left-leader-port COM5` and `--right-leader-port COM6`
- both simulated arms are driven and recorded at the same time

```powershell
python -m lerobot.simulations.bi_so.dataset record ^
  --leader-port COM5 ^
  --repo-id my_user/bi_so_sim_test ^
  --task "Pick and place the cube" ^
  --episode-time-s 30 ^
  --num-episodes 1 ^
  --fps 30 ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco ^
  --launch-viewer
```

Dual-controller example:

```powershell
python -m lerobot.simulations.bi_so.dataset record ^
  --left-leader-port COM5 ^
  --right-leader-port COM6 ^
  --leader-id bimanual_leader ^
  --repo-id my_user/bi_so_dual_sim_test ^
  --root .\outputs\bi_so_dual_sim_test ^
  --task "Pick and place the cube" ^
  --episode-time-s 10 ^
  --num-episodes 1 ^
  --fps 30 ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco ^
  --launch-viewer
```

By default, `record` enables four simulated cameras:
- `left_arm`
- `right_arm`
- `top`
- `front`

Camera behavior:
- `top` is static in the world
- `left_arm` is mounted on the left end-effector
- `right_arm` is mounted on the right end-effector
- `front` is static in the world

You can disable that with:

```powershell
python -m lerobot.simulations.bi_so.dataset record ^
  --no-default-cameras ^
  ...
```

Or override the camera set explicitly:

```powershell
python -m lerobot.simulations.bi_so.dataset record ^
  --camera-names left_arm right_arm top front ^
  ...
```

### 5. Record with two real controllers

`dataset.py record` already supports this directly through `--left-leader-port` and `--right-leader-port`, so you do not need the generic `lerobot_record` path unless you specifically want the broader LeRobot CLI flow.

### 6. Replay a dataset

This also matches the AOSH simulation flow: same script, `replay` mode. Replay sends the full stored action vector back to the simulator, so datasets recorded with one or two controllers both replay correctly.

```powershell
python -m lerobot.simulations.bi_so.dataset replay ^
  --repo-id my_user/bi_so_sim_test ^
  --episode 0 ^
  --fps 30 ^
  --sim-root C:\Users\Ninja\lerobot\src\lerobot\robots\bi_so_follower_simulated\mujoco ^
  --launch-viewer
```

### 7. Edit camera poses

```powershell
python -m lerobot.simulations.bi_so.camera_pose_editor
```

Open directly on one arm-camera XML while still loading the full camera scene:

```powershell
python -m lerobot.simulations.bi_so.camera_pose_editor ^
  --xml-path .\src\lerobot\simulations\bi_so\so_arm100_left_camera.xml
```

### 8. Visualize a recorded episode

```powershell
python -m lerobot.scripts.lerobot_dataset_viz ^
  --repo-id my_user/bi_so_sim_test ^
  --root .\outputs\bi_so_sim_test ^
  --episode-index 0 ^
  --mode local
```

## Startup Behavior

- The startup pose comes from the authored MuJoCo `home` key in the arm XML.
- The local bridge applies that pose to both arms and holds it explicitly.
- Startup pose debugging should go first to:
  - `src/lerobot/robots/bi_so_follower_simulated/mujoco/bridge.py`

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
- record/replay dataset flow
  - look in `dataset.py`
