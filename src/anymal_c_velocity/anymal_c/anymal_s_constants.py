"""ANYmal S (ANYmal C in icosidodecahedron shell) constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

ANYMAL_S_XML: Path = _HERE / "xmls" / "anymal_s.xml"
assert ANYMAL_S_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ANYMAL_S_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ANYMAL_S_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config (same as ANYmal C -- underlying robot is identical).
##

EFFORT_LIMIT = 80.0

# Random small armature since we don't know the real value.
ARMATURE = 0.005

# PD gains derived from armature, targeting 10 Hz natural frequency.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

ANYMAL_S_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HAA", ".*HFE", ".*KFE"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

##
# Keyframes.
# Spawn height raised slightly to account for icosidodecahedron shell radius.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.0),
  joint_pos={
    ".*HAA": 0.0,
    "LF_HFE": 0.4,
    "RF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RH_HFE": -0.4,
    "LF_KFE": -0.8,
    "RF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RH_KFE": 0.8,
    "shell_.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
# Includes both the original robot collision geoms and the shell collision geom.
##

_foot_regex = r"^[LR][FH]_foot$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex, r"^shell_face_\d+$"),
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp={_foot_regex: (0.015, 1, 0.03)},
)

##
# Final config.
##

ANYMAL_S_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ANYMAL_S_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_anymal_s_robot_cfg() -> EntityCfg:
  """Get a fresh ANYmal S robot configuration instance.

  ANYmal S is ANYmal C enclosed in an icosidodecahedron shell.
  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ANYMAL_S_ARTICULATION,
  )


ANYMAL_S_ACTION_SCALE: dict[str, float] = {}
for _a in ANYMAL_S_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _d = _a.damping
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    ANYMAL_S_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_anymal_s_robot_cfg())
  viewer.launch(robot.spec.compile())
