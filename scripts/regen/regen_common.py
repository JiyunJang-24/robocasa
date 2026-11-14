"""Shared helpers for the human-demo regen pipeline.

Builds on scripts/datagen/{dataset_reproduce,reproduce_ur5e_dataset}.py and
scripts/smoke_tests/test_dataset.py (the earlier UR5e replay prototype).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "scripts" / "datagen", REPO_ROOT / "scripts" / "smoke_tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from robosuite.models.grippers import register_gripper  # noqa: E402
from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402
from robosuite.utils.mjcf_utils import xml_path_completion  # noqa: E402

ATOMIC_ROOT = REPO_ROOT / "datasets" / "v1.0" / "pretrain" / "atomic"

FINGER_KP = 40.0
PAD_FRICTION = "2.0 0.05 0.0001"


@register_gripper
class Robotiq85StockGripper(GripperModel):
    """robosuite's stock robotiq_gripper_85.xml (6 joints, 2 position actuators).

    The fork's default Robotiq85Gripper points at the menagerie 2F-85; this keeps
    that untouched and registers the stock model under its own name, with the
    finger servo kp and pad friction used for the regen runs baked into the MJCF
    (so they are also captured in the saved model.xml).
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)
        for act in self.actuator.findall("position"):
            act.set("kp", str(FINGER_KP))
        for geom in self.worldbody.iter("geom"):
            if "fingerpad_collision" in (geom.get("name") or ""):
                geom.set("friction", PAD_FRICTION)

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }

    def format_action(self, action):
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action * np.array([1.0, 1.0]) + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.20

    @property
    def dof(self):
        return 1


from robosuite.models.robots.manipulators.kinova3_robot import Kinova3  # noqa: E402
import robosuite.robots as _rs_robots  # noqa: E402
from robosuite.robots.wheeled_robot import WheeledRobot  # noqa: E402


class Kinova3OmronM(Kinova3):
    """PROVISIONAL Kinova3 on the Omron mobile base (with torso lift), modelled on
    IIWAOmron. lanai's definition was lost; verify against a keti Kinova3 model.xml."""

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Kinova3"}

    @property
    def default_gripper(self):
        return {"right": "Robotiq85StockGripper"}

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


_rs_robots.ROBOT_CLASS_MAPPING.setdefault("Kinova3OmronM", WheeledRobot)


# target robot -> gripper used for the regen
ROBOT_GRIPPERS = {
    "UR5eOmron": "Robotiq85StockGripper",
    "IIWAOmron": "Robotiq140Gripper",
    "Kinova3OmronM": "Robotiq85StockGripper",
}


def resolve_source_dir(task: str) -> Path:
    cands = sorted((ATOMIC_ROOT / task).glob("*/lerobot"))
    cands = [c for c in cands if (c / "extras").is_dir()]
    if not cands:
        raise FileNotFoundError(f"no source lerobot (with extras/) for {task} under {ATOMIC_ROOT}")
    return cands[-1]
