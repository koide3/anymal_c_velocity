"""ANYbotics ANYmal C velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from anymal_c_velocity.anymal_c.anymal_c_constants import (
  ANYMAL_C_ACTION_SCALE,
  get_anymal_c_robot_cfg,
)
from anymal_c_velocity.anymal_c.anymal_s_constants import (
  ANYMAL_S_ACTION_SCALE,
  get_anymal_s_robot_cfg,
)


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
  cfg.observations["actor"].nan_mode = "sanitize"
  cfg.observations["critic"].nan_mode = "sanitize"

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
