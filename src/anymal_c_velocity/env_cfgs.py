"""ANYbotics ANYmal C velocity environment configurations."""

import datetime
import logging

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.nan_guard import NanGuardCfg

from anymal_c_velocity.anymal_c.anymal_c_constants import (
  ANYMAL_C_ACTION_SCALE,
  get_anymal_c_robot_cfg,
)
from anymal_c_velocity.anymal_c.anymal_s_constants import (
  ANYMAL_S_ACTION_SCALE,
  get_anymal_s_robot_cfg,
)

_NAN_LOG_PATH = "/tmp/anymal_log.txt"
_nan_logger = logging.getLogger("anymal_nan_guard")


def _nan_detection_with_logging(env) -> torch.Tensor:
  """Detect NaN/Inf in physics state and log diagnostics on first occurrence."""
  nan_mask = nan_detection(env)
  if not nan_mask.any():
    return nan_mask

  nan_env_ids = torch.where(nan_mask)[0].tolist()
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  lines: list[str] = []
  lines.append(f"\n{'=' * 80}")
  lines.append(f"[{timestamp}] NaN detected in {len(nan_env_ids)} env(s): {nan_env_ids[:20]}")
  lines.append(f"  Episode step: {env.episode_length_buf[nan_env_ids[0]].item()}")
  lines.append(f"  Common step:  {env.common_step_counter}")

  # Physics state summary for first NaN env.
  eid = nan_env_ids[0]
  data = env.sim.data
  for name, tensor in [
    ("qpos", data.qpos),
    ("qvel", data.qvel),
    ("qacc", data.qacc),
    ("sensordata", data.sensordata),
  ]:
    t = tensor[eid]
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    lines.append(
      f"  {name:>14s}: shape={tuple(t.shape)}"
      f"  min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else float('nan'):+.4e}"
      f"  max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else float('nan'):+.4e}"
      f"  NaN={nan_count}  Inf={inf_count}"
    )

  # Reward term values (most recent computation).
  if hasattr(env, "reward_manager"):
    lines.append("  Reward terms:")
    for i, term_name in enumerate(env.reward_manager._term_names):
      cfg = env.reward_manager._term_cfgs[i]
      lines.append(f"    {term_name:>30s}: weight={cfg.weight:.4f}")

  # Observation stats for actor group.
  if hasattr(env, "obs_buf") and "actor" in env.obs_buf:
    obs = env.obs_buf["actor"]
    if isinstance(obs, torch.Tensor):
      o = obs[eid]
      lines.append(
        f"  Actor obs: shape={tuple(o.shape)}"
        f"  NaN={torch.isnan(o).sum().item()}"
        f"  Inf={torch.isinf(o).sum().item()}"
        f"  min={o[torch.isfinite(o)].min().item() if torch.isfinite(o).any() else float('nan'):+.4e}"
        f"  max={o[torch.isfinite(o)].max().item() if torch.isfinite(o).any() else float('nan'):+.4e}"
      )

  # Robot root state.
  try:
    robot = env.scene["robot"]
    root_pos = robot.data.root_link_pos_w[eid]
    root_quat = robot.data.root_link_quat_w[eid]
    root_vel = robot.data.root_com_lin_vel_b[eid]
    lines.append(
      f"  Root pos:  {root_pos.tolist()}"
    )
    lines.append(
      f"  Root quat: {root_quat.tolist()}"
    )
    lines.append(
      f"  Root vel:  {root_vel.tolist()}"
    )
  except Exception:
    lines.append(f"  Root state: <unavailable>")

  lines.append(f"{'=' * 80}\n")

  msg = "\n".join(lines)
  _nan_logger.warning(msg)

  # Append to log file.
  try:
    with open(_NAN_LOG_PATH, "a") as f:
      f.write(msg + "\n")
  except OSError:
    _nan_logger.warning(f"Failed to write NaN log to {_NAN_LOG_PATH}")

  return nan_mask


def anymal_c_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ANYmal C rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 50

  cfg.scene.entities = {"robot": get_anymal_c_robot_cfg()}

  # Set raycast sensor frame to ANYmal C base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base"

  site_names = ("LF", "RF", "LH", "RH")
  geom_names = ("LF_foot", "RF_foot", "LH_foot", "RH_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ANYMAL_C_ACTION_SCALE

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  cfg.rewards["pose"].params["std_standing"] = {
    ".*HAA": 0.05,
    ".*HFE": 0.05,
    ".*KFE": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*HAA": 0.3,
    ".*HFE": 0.3,
    ".*KFE": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    ".*HAA": 0.3,
    ".*HFE": 0.3,
    ".*KFE": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.viz.z_offset = 0.5

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def anymal_c_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ANYmal C flat terrain velocity configuration."""
  cfg = anymal_c_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum.
  cfg.curriculum.pop("terrain_levels", None)

  return cfg


def anymal_s_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ANYmal S (icosidodecahedron shell) rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  # Increase nconmax to handle additional shell-terrain contacts (32 face plates).
  cfg.sim.nconmax = 600

  cfg.scene.entities = {"robot": get_anymal_s_robot_cfg()}

  # Only observe/reward the 12 actuated robot joints (exclude 6 shell joints).
  _rj = (".*HAA", ".*HFE", ".*KFE")
  def _robot_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=_rj)

  # Remove raycast sensor (no height scan needed).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  site_names = ("LF", "RF", "LH", "RH")
  geom_names = ("LF_foot", "RF_foot", "LH_foot", "RH_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Match all collision geoms (including shell_collision).
      pattern=r".*_collision\d*$",
      # Except for the foot geoms and shell struts (shell may touch ground).
      exclude=tuple(geom_names) + tuple(f"shell_face_{i}" for i in range(32)),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
    # Coarsen heightfield to avoid hfield collision overflow with large shell plates.
    for sub_terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
      if hasattr(sub_terrain, "horizontal_scale"):
        sub_terrain.horizontal_scale = 0.5
        # border_width must be >= horizontal_scale.
        if hasattr(sub_terrain, "border_width") and sub_terrain.border_width > 0:
          sub_terrain.border_width = max(sub_terrain.border_width, 0.5)
      if hasattr(sub_terrain, "resolution"):
        sub_terrain.resolution = 0.5

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ANYMAL_S_ACTION_SCALE

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 2.5
  cfg.viewer.elevation = -10.0

  # Filter observations to only include the 12 actuated joints.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["joint_pos"].params["asset_cfg"] = _robot_joints()
    cfg.observations[group].terms["joint_vel"].params["asset_cfg"] = _robot_joints()

  # Remove height_scan observation (not needed with shell).
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Sanitize NaN/Inf in observations (shell physics can diverge).
  cfg.observations["actor"].nan_policy = "sanitize"
  cfg.observations["critic"].nan_policy = "sanitize"

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  # Filter pose reward to only evaluate actuated joints (no shell joints).
  cfg.rewards["pose"].params["asset_cfg"] = _robot_joints()
  cfg.rewards["pose"].params["std_standing"] = {
    ".*HAA": 0.05,
    ".*HFE": 0.05,
    ".*KFE": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*HAA": 0.3,
    ".*HFE": 0.3,
    ".*KFE": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    ".*HAA": 0.3,
    ".*HFE": 0.3,
    ".*KFE": 0.6,
  }

  # Filter dof_pos_limits to actuated joints only.
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = _robot_joints()

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Terminate (and reset) environments with NaN/Inf physics state so that
  # training can continue instead of crashing.
  cfg.terminations["nan_detection"] = TerminationTermCfg(
    func=_nan_detection_with_logging,
    time_out=False,
  )

  # Enable NaN guard to capture simulation states for offline debugging.
  cfg.sim.nan_guard = NanGuardCfg(
    enabled=True,
    buffer_size=100,
    output_dir="/tmp/mjlab/nan_dumps",
    max_envs_to_dump=5,
  )

  cmd = cfg.commands["twist"]
  assert isinstance(cmd, UniformVelocityCommandCfg)
  cmd.viz.z_offset = 0.5

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
