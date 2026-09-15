#!/usr/bin/env python3
"""
Evaluate a LeRobot policy checkpoint on RoboCasa target/test episodes.

Change DEFAULT_CHECKPOINT below, or pass --checkpoint, to evaluate another model.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# REPO_ROOT/scripts/eval/this_file.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# No universal default makes sense for a specific trained checkpoint/dataset --
# these exist so the script is runnable without arguments during development,
# not as something a fresh checkout should expect to find. Always pass
# --checkpoint/--dataset explicitly; the env vars below are for convenience.
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "EVAL_CHECKPOINT",
        REPO_ROOT
        / "outputs"
        / "train"
        / "example"
        / "checkpoints"
        / "050000"
        / "pretrained_model",
    )
)
DEFAULT_DATASET = Path(
    os.environ.get(
        "EVAL_DATASET",
        REPO_ROOT
        / "datasets"
        / "v1.0"
        / "target"
        / "atomic"
        / "TurnOnSinkFaucet"
        / "lerobot",
    )
)
# LeRobot is not vendored here (it is a separate, fast-moving training repo) --
# point this at whatever LeRobot checkout/installation trained the checkpoint.
DEFAULT_LEROBOT_SRC = (
    Path(os.environ["LEROBOT_SRC"]) if "LEROBOT_SRC" in os.environ else None
)
DEFAULT_ROBOSUITE_SRC = Path(
    os.environ.get("ROBOSUITE_SRC", REPO_ROOT / "third_party" / "robosuite")
)
DEFAULT_ROBOCASA_SRC = (
    Path(os.environ["ROBOCASA_X_SRC"]) if "ROBOCASA_X_SRC" in os.environ else None
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("EVAL_OUTPUT_DIR", REPO_ROOT / "outputs" / "eval")
)
DEFAULT_TASK = "Turn on the sink faucet."


LEROBOT_STATE_TO_OBS_KEY = {
    "base_position": "robot0_base_pos",
    "base_rotation": "robot0_base_quat",
    "end_effector_position_relative": "robot0_base_to_eef_pos",
    "end_effector_rotation_relative": "robot0_base_to_eef_quat",
    "gripper_qpos": "robot0_gripper_qpos",
}
ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def _add_local_python_paths(
    lerobot_src: Path, robosuite_src: Path, robocasa_src: Path | None = None
) -> None:
    for path in (robosuite_src, lerobot_src, robocasa_src):
        if path is not None and path.exists():
            sys.path.insert(0, str(path))


def _compat_checkpoint_dir(checkpoint: Path) -> Path:
    """Strip config.json fields this venv's policy dataclass doesn't know.

    Some checkpoints are trained with a newer/forked lerobot (e.g. SmolVLA
    "knowledge insulation": an auxiliary FAST-tokenized action loss,
    stop-gradient'd from the main flow-matching action expert -- training-time
    regularization only). draccus's config parsing rejects any JSON key that
    isn't a declared dataclass field, so a checkpoint saved by that fork's
    SmolVLAConfig (with ki_*/lap_* fields) fails to load in this venv's
    unmodified SmolVLAConfig even though the actual model architecture is
    identical -- confirmed by comparing model.safetensors key sets against a
    known-good checkpoint of this same run family (500/500 keys, no
    fast/ki/lap-named parameters saved). Symlinking everything but a
    keys-filtered config.json into a temp dir lets the standard model load
    exactly as if it had been trained without whatever the fork added.
    """
    import dataclasses
    import tempfile

    config_path = checkpoint / "config.json"
    raw = json.loads(config_path.read_text())
    local_tower = Path(
        "/data1/local/robocasa/outputs/train/all4_n42_all/vision_tower.safetensors"
    )
    if raw.get("type") != "smolvla":
        if (
            raw.get("siglip_encoder_path")
            and not Path(raw["siglip_encoder_path"]).exists()
            and local_tower.exists()
        ):
            tmp_dir = Path(tempfile.mkdtemp(prefix="dp_config_compat_"))
            for item in checkpoint.iterdir():
                if item.name != "config.json":
                    (tmp_dir / item.name).symlink_to(item.resolve())
            raw["siglip_encoder_path"] = str(local_tower)
            (tmp_dir / "config.json").write_text(json.dumps(raw, indent=2))
            return tmp_dir
        return checkpoint
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    known = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
    # Training happened under another checkout; relocate the external SigLIP tower
    # when the recorded absolute path is unavailable on this host.
    tower = raw.get("siglip_encoder_path")
    if tower and not Path(tower).exists() and local_tower.exists():
        raw["siglip_encoder_path"] = str(local_tower)
    extra = sorted(k for k in raw if k not in known)
    if not extra:
        return checkpoint
    print(
        f"[eval] checkpoint config.json has fields this SmolVLAConfig doesn't "
        f"declare (training-only, ignoring for inference): {extra}"
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="smolvla_config_compat_"))
    for item in checkpoint.iterdir():
        if item.name != "config.json":
            (tmp_dir / item.name).symlink_to(item.resolve())
    stripped = {k: v for k, v in raw.items() if k in known}
    (tmp_dir / "config.json").write_text(json.dumps(stripped, indent=2))
    return tmp_dir


def _load_lerobot_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    checkpoint = _compat_checkpoint_dir(checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    __import__(policy_cls.__module__.replace("modeling_", "processor_"))
    policy = policy_cls.from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
    )
    return cfg, policy, preprocessor, postprocessor


class _RandomPolicyConfig:
    def __init__(self, camera_names: list[str]):
        self.input_features = {
            f"observation.images.{camera_name}": type(
                "Feature", (), {"type": type("FeatureType", (), {"name": "VISUAL"})()}
            )()
            for camera_name in camera_names
        }


def _sample_random_env_action(
    env, rng: np.random.Generator, scale: float
) -> np.ndarray:
    action_spec = getattr(env, "action_spec", None)
    if action_spec is None:
        action_spec = env.env.action_spec
    low, high = action_spec
    action = rng.uniform(low=low, high=high).astype(np.float64)
    if scale != 1.0:
        center = (
            np.asarray(low, dtype=np.float64) + np.asarray(high, dtype=np.float64)
        ) / 2.0
        action = center + (action - center) * scale
        action = np.clip(action, low, high)
    return action


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_dataset_metadata(dataset: Path) -> tuple[dict, dict]:
    dataset_meta = _load_json(dataset / "extras" / "dataset_meta.json")
    modality = _load_json(dataset / "meta" / "modality.json")
    return dataset_meta["env_args"], modality


def _override_robot_setup(
    env_meta: dict,
    robot: str | None,
    gripper: str | None,
) -> dict:
    env_meta = deepcopy(env_meta)
    env_kwargs = env_meta.setdefault("env_kwargs", {})
    if robot:
        env_kwargs["robots"] = [robot]
    if gripper and gripper != "dataset":
        env_kwargs["gripper_types"] = (
            None if gripper.lower() in ("none", "null") else gripper
        )
    return env_meta


def _install_robocasa_x_compat_shim() -> None:
    """Allow BARX robocasa_x to load inside the LeRobot eval venv.

    robocasa_x asserts exact mujoco/numpy version strings at import time. The
    eval venv intentionally carries LeRobot's stack instead, so import
    robosuite first while numpy still reports its real version, then satisfy
    robocasa_x's import-time version guards. This is scoped to the eval process.
    """
    import robosuite  # noqa: F401
    import mujoco
    import numpy

    mujoco.__version__ = "3.1.1"
    numpy.__version__ = "1.23.5"

    import robosuite.models.tasks.task as task_module
    import robosuite.models.base as model_base
    import robosuite.utils.binding_utils as binding_utils

    def safe_get_ids(sim, elements, element_type, inplace=False):
        if type(elements) is not list:
            elements = [elements]
        if not inplace:
            elements = list(elements)
        out = []
        for element in elements:
            if type(element) is int:
                out.append(element)
                continue
            element_func = sim.model.__getattribute__(f"{element_type}_name2id")
            try:
                out.append(element_func(element))
            except ValueError:
                continue
        if inplace:
            elements[:] = out
            return elements
        return out

    task_module.get_ids = safe_get_ids

    def compat_generate_id_mappings(self, sim):
        self._instances_to_ids = {}
        self._classes_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}

        models = [model for model in self.mujoco_objects]
        for robot in self.mujoco_robots:
            models += [robot] + robot.models

        worldbody = self.mujoco_arena.root.find("worldbody")
        exclude_bodies = {"table", "left_eef_target", "right_eef_target"}
        top_level_bodies = [
            body.attrib.get("name")
            for body in worldbody.findall("body")
            if body.attrib.get("name") not in exclude_bodies
        ]
        models.extend(top_level_bodies)

        for model in models:
            if isinstance(model, str):
                body_name = model
                visual_group_number = 1
                try:
                    body_id = sim.model.body_name2id(body_name)
                except ValueError:
                    continue
                inst, cls = body_name, body_name
                geom_ids = task_module.get_subtree_geom_ids_by_group(
                    sim.model, body_id, visual_group_number
                )
                id_groups = [geom_ids, []]
            else:
                cls = str(type(model)).split("'")[1].split(".")[-1]
                inst = model.name
                id_groups = [
                    safe_get_ids(
                        sim=sim,
                        elements=model.visual_geoms + model.contact_geoms,
                        element_type="geom",
                    ),
                    safe_get_ids(sim=sim, elements=model.sites, element_type="site"),
                ]

            group_types = ("geom", "site")
            ids_to_instances = (
                self._geom_ids_to_instances,
                self._site_ids_to_instances,
            )
            ids_to_classes = (self._geom_ids_to_classes, self._site_ids_to_classes)

            if inst in self._instances_to_ids:
                continue
            self._instances_to_ids[inst] = {}
            if cls not in self._classes_to_ids:
                self._classes_to_ids[cls] = {
                    group_type: [] for group_type in group_types
                }

            for ids, group_type, ids_to_inst, ids_to_cls in zip(
                id_groups, group_types, ids_to_instances, ids_to_classes
            ):
                unique_ids = []
                for idn in ids:
                    if idn in ids_to_inst:
                        continue
                    unique_ids.append(idn)
                    ids_to_inst[idn] = inst
                    ids_to_cls[idn] = cls
                self._instances_to_ids[inst][group_type] = unique_ids
                self._classes_to_ids[cls][group_type] += unique_ids

    task_module.Task.generate_id_mappings = compat_generate_id_mappings

    def safe_set_sites_visibility(self, sim, visible):
        for vis_g in self.sites:
            try:
                vis_g_id = sim.model.site_name2id(vis_g)
            except ValueError:
                continue
            if (visible and sim.model.site_rgba[vis_g_id][3] < 0) or (
                not visible and sim.model.site_rgba[vis_g_id][3] > 0
            ):
                sim.model.site_rgba[vis_g_id][3] = -sim.model.site_rgba[vis_g_id][3]

    model_base.MujocoModel.set_sites_visibility = safe_set_sites_visibility

    original_site_name2id = binding_utils.MjModel.site_name2id
    original_joint_name2id = binding_utils.MjModel.joint_name2id
    original_get_site_xpos = binding_utils.MjData.get_site_xpos

    def compat_name_aliases(name):
        aliases = []
        stems = [name]
        if name.startswith("stovetop_"):
            stems.append("stove_" + name.removeprefix("stovetop_"))
        if name.startswith("stove_"):
            stems.append("stovetop_" + name.removeprefix("stove_"))

        group_tokens = (
            "_main_group_",
            "_left_group_",
            "_right_group_",
            "_front_group_",
        )
        for stem in stems:
            if stem not in aliases:
                aliases.append(stem)
            for group in group_tokens:
                if group not in stem:
                    continue
                for replacement in group_tokens:
                    alias = stem.replace(group, replacement, 1)
                    if alias not in aliases:
                        aliases.append(alias)

        for alias in list(aliases):
            if alias.endswith("_coffee_liquid_left") or alias.endswith(
                "_coffee_liquid_right"
            ):
                aliases.append(alias.rsplit("_", 1)[0])
        return [alias for alias in aliases if alias != name]

    def normalized_compat_name(name):
        if name.startswith("stovetop_"):
            name = "stove_" + name.removeprefix("stovetop_")
        for group in ("main", "left", "right", "front"):
            name = name.replace(f"_{group}_group_", "_group_")
        if name.endswith("_coffee_liquid_left") or name.endswith(
            "_coffee_liquid_right"
        ):
            name = name.rsplit("_", 1)[0]
        return name

    def find_normalized_name(name, names):
        normalized_name = normalized_compat_name(name)
        matches = [
            candidate
            for candidate in names
            if normalized_compat_name(candidate) == normalized_name
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def compat_site_name2id(self, name):
        try:
            return original_site_name2id(self, name)
        except ValueError:
            for alias in compat_name_aliases(name):
                try:
                    return original_site_name2id(self, alias)
                except ValueError:
                    pass
            normalized_match = find_normalized_name(
                name, getattr(self, "site_names", ())
            )
            if normalized_match is not None:
                return original_site_name2id(self, normalized_match)
            raise

    def compat_joint_name2id(self, name):
        try:
            return original_joint_name2id(self, name)
        except ValueError:
            for alias in compat_name_aliases(name):
                try:
                    return original_joint_name2id(self, alias)
                except ValueError:
                    pass
            normalized_match = find_normalized_name(
                name, getattr(self, "joint_names", ())
            )
            if normalized_match is not None:
                return original_joint_name2id(self, normalized_match)
            raise

    binding_utils.MjModel.site_name2id = compat_site_name2id
    binding_utils.MjModel.joint_name2id = compat_joint_name2id

    def compat_get_site_xpos(self, name):
        try:
            return original_get_site_xpos(self, name)
        except ValueError:
            bounds_suffixes = (
                "_ext_p0",
                "_ext_px",
                "_ext_py",
                "_ext_pz",
                "_int_p0",
                "_int_px",
                "_int_py",
                "_int_pz",
            )
            if name.endswith(bounds_suffixes):
                return np.zeros(3)
            raise

    binding_utils.MjData.get_site_xpos = compat_get_site_xpos


def _episode_paths(
    dataset: Path, episodes: list[int] | None, limit: int | None
) -> list[Path]:
    paths = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
    if episodes is not None:
        wanted = {f"episode_{ep:06d}.parquet" for ep in episodes}
        paths = [p for p in paths if p.name in wanted]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset}")
    return paths


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _episode_payload(
    dataset: Path, episode_index: int
) -> tuple[np.ndarray, dict, str | None]:
    episode_dir = dataset / "extras" / f"episode_{episode_index:06d}"
    states = np.load(episode_dir / "states.npz", allow_pickle=True)["states"]
    ep_meta = _load_json(episode_dir / "ep_meta.json")
    model_path = episode_dir / "model.xml.gz"
    model_xml = None
    if model_path.exists():
        with gzip.open(model_path, "rt") as f:
            model_xml = f.read()
    return states, ep_meta, model_xml


def _camera_names_from_features(input_features: dict) -> list[str]:
    camera_names = []
    for key, feature in input_features.items():
        feature_type = getattr(feature.type, "name", str(feature.type))
        if feature_type == "VISUAL" and key.startswith("observation.images."):
            camera_names.append(key.removeprefix("observation.images."))
    return camera_names


def _camera_resolution_from_features(input_features: dict) -> tuple[int, int]:
    """(height, width) the checkpoint's own visual features were trained at.

    Datasets don't all share one resolution -- atomic_new's cameras are
    180x320 (preserving the mg recordings' native aspect ratio), while the
    older barx atomic datasets are square 256x256. A checkpoint trained on
    one and evaluated by rendering the other doesn't just look softer: the
    aspect ratio changes the field of view actually captured (180x320 spans
    ~72.6deg horizontally at fovy=45, a square 256x256 crop only ~45deg), so
    the policy sees a genuinely different framing than it was trained on.
    Falls back to 256 (the old hardcoded value) if no visual feature carries
    a shape -- e.g. --random-actions mode, whose synthetic input_features
    only declare a type.
    """
    for feature in input_features.values():
        feature_type = getattr(feature.type, "name", str(feature.type))
        if feature_type != "VISUAL":
            continue
        shape = getattr(feature, "shape", None)
        if shape and len(shape) == 3:
            _, h, w = shape
            return int(h), int(w)
    return 256, 256


def _make_env(
    env_meta: dict,
    seed: int,
    camera_names: list[str],
    camera_height: int = 256,
    camera_width: int = 256,
):
    import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    env_meta = deepcopy(env_meta)
    _resolve_env_name(env_meta)
    env_kwargs = env_meta.setdefault("env_kwargs", {})
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["use_object_obs"] = True
    env_kwargs["camera_depths"] = False
    env_kwargs["camera_names"] = camera_names
    env_kwargs["camera_widths"] = camera_width
    env_kwargs["camera_heights"] = camera_height
    env_kwargs["translucent_robot"] = False

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={
            "obs": {
                "low_dim": [],
                "rgb": [f"{camera_name}_image" for camera_name in camera_names],
            }
        }
    )
    # robocasa_x only knows the "A"/"B" object splits and raises on anything
    # else; robocasa v1.0 datasets say "target". The split only picks which
    # object instances a freshly sampled scene may draw from, and no evaluated
    # scene is freshly sampled, so normalise it away.
    split = env_kwargs.get("obj_instance_split")
    if split is not None and split not in ("A", "B"):
        print(
            f"[eval] obj_instance_split={split!r} unknown to this robocasa; using the full set"
        )
        env_kwargs["obj_instance_split"] = None

    # robocasa v1.0 datasets carry env kwargs this build's Kitchen no longer
    # takes (clutter_mode, ...). They only affect how a *fresh* scene would be
    # sampled; every episode here is rebuilt from its stored model XML and
    # ep_meta, so dropping them does not change what is evaluated.
    dropped = []
    while True:
        try:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
                seed=seed,
            )
            break
        except TypeError as exc:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if m is None or m.group(1) not in env_kwargs:
                raise
            env_kwargs.pop(m.group(1))
            dropped.append(m.group(1))
    if dropped:
        print(f"[eval] dropped env kwargs unsupported by this robocasa: {dropped}")
    env.postprocess_visual_obs = False
    return env


def _install_mesh_sanitizer() -> None:
    """Route every MuJoCo compile in robosuite through `_sanitize_episode_xml`.

    Episode XMLs were already sanitised explicitly, but the *env constructor*
    compiles a freshly sampled scene of its own, and robocasa v1.0 datasets
    sample scenes whose meshes trip MuJoCo's inertial compilation under this
    venv. Wrapping the compile entry point covers both without each call site
    having to remember.
    """
    from robosuite.utils import binding_utils

    if getattr(binding_utils.MjSim, "_eval_mesh_sanitizer", False):
        return
    original = binding_utils.MjSim.from_xml_string.__func__

    def from_xml_string(cls, xml, *args, **kwargs):
        # Must happen here rather than at the call sites: robosuite runs
        # `_xml_processors` (which re-point asset paths at the active robocasa
        # install) between any sanitising we do and the actual compile.
        xml = _repair_xml_assets(xml)
        try:
            return original(cls, xml, *args, **kwargs)
        except ValueError as exc:
            if "mesh volume is too small" not in str(exc):
                raise
            return original(cls, _sanitize_episode_xml(xml), *args, **kwargs)

    binding_utils.MjSim.from_xml_string = classmethod(from_xml_string)
    binding_utils.MjSim._eval_mesh_sanitizer = True


def _resolve_env_name(env_meta: dict) -> None:
    """Rename a dataset's env to whatever this robocasa build registers.

    robocasa v1.0 datasets record `PickPlaceCounterToStove`; robocasa_x
    registers the same task as `PnPCounterToStove`. Rewrite in place so those
    datasets can be evaluated without touching the dataset itself.
    """
    from robosuite.environments.base import REGISTERED_ENVS

    registered = {getattr(e, "__name__", str(e)) for e in REGISTERED_ENVS}
    name = env_meta.get("env_name") or env_meta.get("env_kwargs", {}).get("env_name")
    if not name or name in registered:
        return
    # PickPlaceCounterToStove -> PnPCounterToStove, and the reverse
    alts = []
    if name.startswith("PickPlace"):
        alts.append("PnP" + name[len("PickPlace") :])
    elif name.startswith("PnP"):
        alts.append("PickPlace" + name[len("PnP") :])
    alts += ["X" + a for a in list(alts) + [name]]
    for alt in alts:
        if alt in registered:
            print(f"[eval] env '{name}' is not registered; using '{alt}'")
            env_meta["env_name"] = alt
            if "env_name" in env_meta.get("env_kwargs", {}):
                env_meta["env_kwargs"]["env_name"] = alt
            return
    raise ValueError(
        f"env '{name}' is not registered and no alias matched " f"(tried {alts})"
    )


def _install_episode_meta(env, ep_meta: dict) -> None:
    if hasattr(env.env, "set_attrs_from_ep_meta"):
        env.env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env.env, "set_ep_meta"):
        env.env.set_ep_meta(ep_meta)


def _reset_to_episode(env, state: np.ndarray, ep_meta: dict, model_xml: str | None):
    _install_episode_meta(env, ep_meta)
    reset_state = {"states": np.asarray(state).reshape(-1)}
    if model_xml is not None:
        return _reset_to_episode_xml_without_prereset(
            env, reset_state["states"], ep_meta, model_xml
        )
    env.reset()
    return env.reset_to(reset_state)


def _install_recorded_camera_override(raw_env, ep_meta: dict) -> None:
    """Point every camera this episode recorded at its actual recorded pose.

    atomic_new was generated by `convert_mg_to_atomic_new.py`'s
    `install_front_camera_override`, which hijacks `Kitchen.set_cameras` so
    `robot0_agentview_right` gets the mg render's own per-episode-randomised
    `<robot>_agentview_left` pose instead of RoboCasa's generic default --
    that final pose is what ends up saved in this episode's own
    `ep_meta["cam_configs"]`. `_reset_to_episode_xml_without_prereset` never
    triggers `_load_model`/`set_cameras` again (it loads the stored XML via
    `reset_from_xml_string`, which -- with `deterministic_reset=True` --
    reuses `self.sim.reset()` instead), so without this, `edit_model_xml`
    injects cameras from whatever generic `self._cam_configs` the env's
    *first* random construction produced. That silently renders a completely
    different, unrelated viewpoint (confirmed: mean pixel diff ~58/255,
    max 237 against the dataset's own recorded video) while still succeeding
    with the right image shape, so nothing about it looks broken until you
    diff it against the source video.
    """
    cam_configs = ep_meta.get("cam_configs")
    if not cam_configs or not hasattr(raw_env, "_cam_configs"):
        return
    for cam_name, cam_cfg in cam_configs.items():
        raw_env._cam_configs[cam_name] = deepcopy(cam_cfg)


def _reset_to_episode_xml_without_prereset(
    env, state: np.ndarray, ep_meta: dict, model_xml: str
):
    """Load an episode XML directly, bypassing EnvRobocasa.reset_to's pre-reset.

    BARX robocasa_x episodes already store the exact MJCF used for replay. In
    the LeRobot eval venv, the wrapper's preliminary env.reset() can fail while
    compiling a freshly generated random scene under newer MuJoCo, before the
    stored episode XML is ever loaded. Loading the stored XML directly avoids
    that unrelated compilation path.
    """
    raw_env = env.env
    _install_episode_meta(env, ep_meta)
    _install_recorded_camera_override(raw_env, ep_meta)
    xml = (
        raw_env.edit_model_xml(model_xml)
        if hasattr(raw_env, "edit_model_xml")
        else model_xml
    )
    xml = _sanitize_episode_xml(xml)

    # Honour --robot / --gripper even though the episode XML carries the
    # dataset's own. Arm first (coarser), then gripper on top of it.
    swapped = False
    pre_xml = xml
    robot_donor = getattr(env, "_robot_donor_xml", None)
    env._arm_swapped = False
    if robot_donor is not None and _robot_signature(robot_donor) != _robot_signature(
        xml
    ):
        xml = _sanitize_episode_xml(_swap_robot_in_xml(xml, robot_donor))
        swapped = True
        env._arm_swapped = True
    donor = getattr(env, "_gripper_donor_xml", None)
    if donor is not None and _gripper_signature(donor) != _gripper_signature(xml):
        xml = _sanitize_episode_xml(_swap_gripper_in_xml(xml, donor))
        swapped = True

    _filter_model_geom_refs_to_xml(raw_env.model, xml)
    _align_robot_gripper_joints_to_xml(raw_env, xml)
    raw_env.reset_from_xml_string(xml)
    _refresh_fixture_poses(raw_env, xml)
    raw_env.sim.reset()
    if swapped:
        # the flat state was recorded against the original gripper's layout
        _set_state_by_joint_name(raw_env, pre_xml, np.asarray(state).reshape(-1))
        if getattr(env, "_arm_swapped", False):
            ref = _recorded_eef_pose(pre_xml, np.asarray(state).reshape(-1))
            if ref is not None:
                err = _ik_arm_to_pose(raw_env, ref[0], ref[1])
                if err is not None and err > 5e-3:
                    print(f"[warn] arm IK left {err * 1000:.1f}mm of eef error")
        donor_qpos = getattr(env, "_gripper_donor_qpos", None)
        if donor_qpos:
            model, data = raw_env.sim.model, raw_env.sim.data
            for name, value in donor_qpos.items():
                jid = model.joint_name2id(name) if name in model.joint_names else -1
                if jid >= 0:
                    data.qpos[model.jnt_qposadr[jid]] = value
            raw_env.sim.forward()
    else:
        raw_env.sim.set_state_from_flattened(state)
    raw_env.sim.forward()
    if hasattr(raw_env, "update_sites"):
        raw_env.update_sites()
    if hasattr(raw_env, "update_state"):
        raw_env.update_state()
    return env.get_observation()


def _gripper_signature(xml: str) -> frozenset:
    """Identify which gripper model an XML carries.

    Body names, not joint names: robosuite's two Robotiq 85 models and the
    Robotiq 140 all use the same joint names (`finger_joint`,
    `*_inner_finger_joint`, ...), so joints alone cannot tell them apart and a
    swap would be silently skipped. The body trees do differ
    (`robotiq_85_adapter_link` vs `base_mount` vs `right_gripper`).
    """
    root = ET.fromstring(xml) if isinstance(xml, str) else xml
    return frozenset(
        b.get("name")
        for b in root.iter("body")
        if (b.get("name") or "").startswith("gripper0_")
    ) | frozenset(
        j.get("name")
        for j in root.iter("joint")
        if (j.get("name") or "").startswith("gripper0_")
    )


def _register_legacy_robotiq85() -> str:
    """Register the Robotiq 85 model the BARX datasets were recorded with.

    robosuite's `Robotiq85Gripper` now points at the Menagerie 4-bar model
    (`robotiq_2f85_menagerie/2f85_robosuite.xml`); the recorded episodes use the
    older `robotiq_gripper_85.xml`, whose grip_site sits 11mm closer and rotated
    90deg. Transplanting the recorded gripper subtree onto a *different* arm is
    not enough -- the mount pos/quat in that subtree belongs to the arm it was
    recorded on, so on another wrist the gripper hangs off to one side. Letting
    robosuite mount this XML itself puts it on whatever wrist it is given, with
    that robot's own offsets.
    """
    import numpy as _np
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.models.grippers.robotiq_85_gripper import Robotiq85Gripper
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.utils.mjcf_utils import xml_path_completion

    name = "Robotiq85GripperLegacy"
    if name in GRIPPER_MAPPING:
        return name

    class Robotiq85GripperLegacy(Robotiq85Gripper):
        def __init__(self, idn=0):
            GripperModel.__init__(
                self, xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn
            )

        @property
        def init_qpos(self):
            return _np.zeros(6)

        @property
        def _important_geoms(self):
            return {
                "left_finger": ["left_inner_finger_collision"],
                "right_finger": ["right_inner_finger_collision"],
                "left_fingerpad": ["left_fingerpad_collision"],
                "right_fingerpad": ["right_fingerpad_collision"],
            }

    Robotiq85GripperLegacy.__name__ = name
    GRIPPER_MAPPING[name] = Robotiq85GripperLegacy
    return name


def _robot_signature(xml: str) -> frozenset:
    root = ET.fromstring(xml) if isinstance(xml, str) else xml
    return frozenset(
        j.get("name")
        for j in root.iter("joint")
        if (j.get("name") or "").startswith("robot0_")
    )


def _swap_robot_in_xml(stored_xml: str, donor_xml: str) -> str:
    """Put the donor's arm (and whatever gripper it carries) on the stored scene.

    Same idea as `_swap_gripper_in_xml`, one level up the tree: the arm hangs
    off `mobilebase0_support` as a single `robot0_*` body, so the whole
    manipulator can be transplanted while the kitchen, the mobile base and the
    object placements stay exactly as recorded. Needed because there is no
    JacoOmron dataset -- the only way to evaluate that arm on these episodes is
    to graft it onto another robot's recording.

    The arm's joints do not survive (different kinematics), so it starts from
    the donor's home pose rather than the recorded one.
    """
    stored = ET.fromstring(stored_xml)
    donor = ET.fromstring(donor_xml)

    def mount(root):
        for b in root.iter("body"):
            if b.get("name") == "mobilebase0_support":
                return b
        raise ValueError("mobilebase0_support not found")

    s_mount, d_mount = mount(stored), mount(donor)
    # Match on position in the tree, not on name: the arm root is whatever
    # single body hangs off the torso, and it is *not* named consistently
    # (recorded scenes call it `robot0_fixed_base_link`, freshly built ones
    # `manipulator_mount`). Filtering by a `robot0_` prefix silently copied
    # nothing while still dragging the donor's gripper contact pairs across,
    # leaving dangling body references that fail to compile.
    incoming = [copy.deepcopy(c) for c in d_mount if c.tag == "body"]

    # The arm carries its own cameras (robotview, eye_in_hand). The recorded
    # scene already defines cameras of the same name, and MuJoCo rejects
    # duplicates, so drop the stored copies and let the transplanted arm's win
    # -- a wrist camera has to follow the wrist it is bolted to.
    incoming_cams = {
        c.get("name") for b in incoming for c in b.iter("camera") if c.get("name")
    }
    if incoming_cams:
        parents = {child: parent for parent in stored.iter() for child in parent}
        for cam in [c for c in stored.iter("camera") if c.get("name") in incoming_cams]:
            parent = parents.get(cam)
            if parent is not None:
                parent.remove(cam)

    for child in [c for c in s_mount if c.tag == "body"]:
        s_mount.remove(child)
    for child in incoming:
        s_mount.append(child)

    def is_robot(el) -> bool:
        blob = " ".join(str(v) for v in el.attrib.values())
        blob += " " + " ".join(str(v) for c in el.iter() for v in c.attrib.values())
        return "robot0_" in blob or "gripper0_" in blob

    for section in ("tendon", "equality", "actuator", "contact", "sensor"):
        s_sec, d_sec = stored.find(section), donor.find(section)
        if s_sec is not None:
            for el in [e for e in s_sec if is_robot(e)]:
                s_sec.remove(el)
        if d_sec is not None:
            keep = [e for e in d_sec if is_robot(e)]
            if keep:
                if s_sec is None:
                    s_sec = ET.SubElement(stored, section)
                for el in keep:
                    s_sec.append(copy.deepcopy(el))

    _merge_assets(stored, donor)
    return ET.tostring(stored, encoding="unicode")


def _merge_assets(stored, donor) -> None:
    """Copy over any donor asset the (already transplanted) bodies now refer to."""
    d_asset = donor.find("asset")
    if d_asset is None:
        return
    s_asset = stored.find("asset")
    if s_asset is None:
        s_asset = ET.SubElement(stored, "asset")
    have = {(e.tag, e.get("name")) for e in s_asset}
    used = set()
    for el in stored.iter():
        for key in ("mesh", "material", "texture"):
            if el.get(key):
                used.add(el.get(key))
    for el in d_asset:
        name = el.get("name")
        if name and name in used and (el.tag, name) not in have:
            s_asset.append(copy.deepcopy(el))
            have.add((el.tag, name))
    for _ in range(2):  # materials pull in textures
        tex = {e.get("texture") for e in s_asset if e.get("texture")}
        for el in d_asset:
            if (
                el.tag == "texture"
                and el.get("name") in tex
                and (el.tag, el.get("name")) not in have
            ):
                s_asset.append(copy.deepcopy(el))
                have.add((el.tag, el.get("name")))


def _swap_gripper_in_xml(stored_xml: str, donor_xml: str) -> str:
    """Put the donor's gripper onto the stored episode's scene.

    `--gripper` only affects the model robosuite builds at env construction;
    every episode then loads its recorded XML, which carries the gripper the
    dataset was generated with, so the override was silently reverted. Rebuilding
    the scene from ep_meta instead is not an option here -- for these BARX
    datasets it yields a *different kitchen* (different cabinets/fridge), so the
    episode would no longer be the episode.

    Instead keep the recorded scene byte-for-byte and transplant just the
    gripper, which robosuite localises nicely: one body under
    `robot0_right_hand`, plus its tendon/equality/actuator entries and assets,
    all under the `gripper0_` prefix. Donor and stored use identical naming
    because robosuite generated both the same way.
    """
    stored = ET.fromstring(stored_xml)
    donor = ET.fromstring(donor_xml)

    def hand(root):
        for b in root.iter("body"):
            if b.get("name") == "robot0_right_hand":
                return b
        raise ValueError("robot0_right_hand not found")

    s_hand, d_hand = hand(stored), hand(donor)
    for child in [
        c
        for c in s_hand
        if c.tag == "body" and (c.get("name") or "").startswith("gripper0_")
    ]:
        s_hand.remove(child)
    for child in d_hand:
        if child.tag == "body" and (child.get("name") or "").startswith("gripper0_"):
            s_hand.append(copy.deepcopy(child))

    def is_gripper(el) -> bool:
        blob = " ".join(str(v) for v in el.attrib.values())
        blob += " " + " ".join(str(v) for c in el.iter() for v in c.attrib.values())
        return "gripper0_" in blob

    for section in ("tendon", "equality", "actuator", "contact", "sensor"):
        s_sec = stored.find(section)
        d_sec = donor.find(section)
        if s_sec is not None:
            for el in [e for e in s_sec if is_gripper(e)]:
                s_sec.remove(el)
        if d_sec is not None:
            keep = [e for e in d_sec if is_gripper(e)]
            if keep:
                if s_sec is None:
                    s_sec = ET.SubElement(stored, section)
                for el in keep:
                    s_sec.append(copy.deepcopy(el))

    # assets the transplanted gripper refers to (meshes/materials/textures)
    s_asset = stored.find("asset")
    d_asset = donor.find("asset")
    if d_asset is not None:
        if s_asset is None:
            s_asset = ET.SubElement(stored, "asset")
        have = {(e.tag, e.get("name")) for e in s_asset}
        used = set()
        for el in stored.iter():
            for key in ("mesh", "material", "texture"):
                if el.get(key):
                    used.add(el.get(key))
        for el in d_asset:
            name = el.get("name")
            if name and name in used and (el.tag, name) not in have:
                s_asset.append(copy.deepcopy(el))
                have.add((el.tag, name))
        # materials pull in textures
        for _ in range(2):
            tex = {e.get("texture") for e in s_asset if e.get("texture")}
            for el in d_asset:
                if (
                    el.tag == "texture"
                    and el.get("name") in tex
                    and (el.tag, el.get("name")) not in have
                ):
                    s_asset.append(copy.deepcopy(el))
                    have.add((el.tag, el.get("name")))
    return ET.tostring(stored, encoding="unicode")


def _donor_gripper_qpos(donor_xml: str, donor_state: np.ndarray) -> dict:
    """Gripper joint pose recorded alongside the donor gripper, by joint name.

    A transplanted gripper otherwise starts at the model's qpos0 (all zeros),
    which for these Robotiq linkages is not a configuration the coupling
    tendons can hold -- the fingers snap to nonsense on the first step (joints
    jumping to >2 rad) and the gripper renders visibly broken. Seeding it with
    the pose the donor dataset actually recorded gives a consistent starting
    configuration.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(_sanitize_episode_xml(donor_xml))
    qpos = np.asarray(donor_state).reshape(-1)[1 : 1 + model.nq]
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name.startswith("gripper0_"):
            out[name] = float(qpos[model.jnt_qposadr[j]])
    return out


def _recorded_eef_pose(old_xml: str, state: np.ndarray):
    """World pose of the grip site in the episode's own recording."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(_sanitize_episode_xml(old_xml))
    data = mujoco.MjData(model)
    flat = np.asarray(state).reshape(-1)
    data.qpos[:] = flat[1 : 1 + model.nq]
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
    if sid < 0:
        return None
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def _arm_init_qpos(raw_env, n_joints: int):
    """The arm's natural home pose, as robosuite would set it on a normal reset.

    Needed because `_ik_arm_to_pose` runs right after `sim.reset()`, where the
    arm is at the model's qpos0 rather than its home pose. For Jaco those are
    very different: qpos0 is the fully-extended, singular configuration.
    """
    robots = getattr(raw_env, "robots", None)
    if not robots:
        return None
    robot = robots[0]
    for holder in (robot, getattr(robot, "robot_model", None)):
        if holder is None:
            continue
        try:
            q = np.asarray(getattr(holder, "init_qpos"), dtype=float).reshape(-1)
        except (AttributeError, TypeError, ValueError):
            continue
        if q.size == n_joints:
            return q
    return None


def _ik_arm_to_pose(
    raw_env,
    target_pos,
    target_mat,
    restarts: int = 12,
    iters: int = 200,
    damping: float = 1e-2,
):
    """Put a transplanted arm where the recording's gripper was.

    A different arm cannot inherit the recorded joint angles, so it would
    otherwise start at its own home pose -- for Jaco that is folded down beside
    the base, nowhere near the counter, which makes the episode unwinnable
    before the policy even acts.

    Plain damped least-squares is not enough on its own: Jaco is 7-DOF with
    +-360deg on four joints, so there is an enormous redundant space and a
    single solve happily lands in a folded, self-intersecting branch that
    reaches the target but looks (and behaves) broken. So try several seeds and
    score the results on self-collision first, then accuracy, then distance
    from the arm's natural posture.
    """
    import mujoco

    sim = raw_env.sim
    wm, wd = sim.model, sim.data
    m = wm._model if hasattr(wm, "_model") else wm
    d = wd._data if hasattr(wd, "_data") else wd

    names = [
        n
        for n in wm.joint_names
        if n
        and n.startswith("robot0_")
        and m.jnt_type[wm.joint_name2id(n)] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if not names:
        return None
    jids = np.array([wm.joint_name2id(n) for n in names])
    qadr, dadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    sid = wm.site_name2id("gripper0_right_grip_site")
    robot_geoms = {
        wm.geom_name2id(n)
        for n in wm.geom_names
        if n and (n.startswith("robot0_") or n.startswith("gripper0_"))
    }
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    # NOT d.qpos: robosuite injects init_qpos during its own reset, not into the
    # model, so straight after sim.reset() the arm sits at qpos0 -- for Jaco that
    # is fully extended and singular. Seeding and nullspace-biasing towards that
    # is what produced the stretched-out arm with the elbow off in space.
    q_home = _arm_init_qpos(raw_env, len(qadr))
    if q_home is None:
        q_home = np.array(d.qpos[qadr], dtype=float)
    q_home = np.clip(q_home, lo, hi)
    eye = np.eye(len(qadr))

    def errors():
        pos_err = target_pos - d.site_xpos[sid]
        rot = target_mat @ d.site_xmat[sid].reshape(3, 3).T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot, dtype=float).reshape(9))
        n_ax = np.linalg.norm(quat[1:])
        angle = 2 * np.arctan2(n_ax, quat[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        return pos_err, (quat[1:] / n_ax * angle) if n_ax > 1e-9 else np.zeros(3)

    def self_collisions():
        mujoco.mj_forward(m, d)
        return sum(
            1
            for i in range(d.ncon)
            if d.contact[i].geom1 in robot_geoms and d.contact[i].geom2 in robot_geoms
        )

    # Measure the scene's baseline contacts at the *home* pose, before any IK.
    # Previously this was taken from attempt 0's solution, so if that solution
    # self-intersected the collision term was calibrated away to zero.
    d.qpos[qadr] = q_home
    d.qvel[:] = 0.0
    base_collisions = self_collisions()

    rng = np.random.default_rng(0)
    best = None
    for attempt in range(restarts):
        if attempt == 0:
            d.qpos[qadr] = q_home
        else:
            d.qpos[qadr] = np.clip(q_home + rng.normal(0, 0.35, len(qadr)), lo, hi)
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            pos_err, rot_err = errors()
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            jac = np.vstack([jacp[:, dadr], jacr[:, dadr]])
            pinv = jac.T @ np.linalg.solve(
                jac @ jac.T + damping**2 * np.eye(6), np.eye(6)
            )
            home_err = (q_home - np.array(d.qpos[qadr], dtype=float) + np.pi) % (
                2 * np.pi
            ) - np.pi
            dq = pinv @ np.concatenate([pos_err, rot_err]) + (eye - pinv @ jac) @ (
                0.05 * home_err
            )
            d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lo, hi)

        pos_err, rot_err = errors()
        ncol = self_collisions()
        drift = float(
            np.linalg.norm(
                (q_home - np.array(d.qpos[qadr]) + np.pi) % (2 * np.pi) - np.pi
            )
        )
        # Units: position in mm. Drift at 8/rad means a contorted branch has to
        # buy at least 8mm of accuracy per radian of extra joint travel to win,
        # which keeps the elbow near the arm's natural posture instead of
        # swinging it out of the workspace.
        score = (
            max(ncol - base_collisions, 0) * 25.0
            + np.linalg.norm(pos_err) * 1000.0
            + np.degrees(np.linalg.norm(rot_err)) * 0.5
            + drift * 8.0
        )
        if best is None or score < best[0]:
            best = (
                score,
                np.array(d.qpos[qadr], dtype=float),
                float(np.linalg.norm(pos_err)),
                ncol,
            )
        if (
            np.linalg.norm(pos_err) < 1e-3
            and ncol <= base_collisions
            and np.degrees(np.linalg.norm(rot_err)) < 5
            and drift < 2.5
        ):
            break

    d.qpos[qadr] = best[1]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return best[2]


def _set_state_by_joint_name(raw_env, old_xml: str, state: np.ndarray) -> None:
    """Restore a recorded state onto a model whose joint layout changed.

    Swapping the gripper changes nq/nv, so the flat state vector no longer lines
    up. Every non-gripper joint is shared though, so copy per joint by name and
    leave the new gripper at its default pose.
    """
    import mujoco

    old = mujoco.MjModel.from_xml_string(old_xml)
    nq_old = old.nq
    qpos_old, qvel_old = state[1 : 1 + nq_old], state[1 + nq_old : 1 + nq_old + old.nv]
    new = raw_env.sim.model
    data = raw_env.sim.data
    sizes = {
        mujoco.mjtJoint.mjJNT_FREE: (7, 6),
        mujoco.mjtJoint.mjJNT_BALL: (4, 3),
        mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
        mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
    }
    old_idx = {}
    for j in range(old.njnt):
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j)
        old_idx[name] = (old.jnt_qposadr[j], old.jnt_dofadr[j], old.jnt_type[j])
    moved = 0
    for j in range(new.njnt):
        name = new.joint_names[j]
        if name not in old_idx or name.startswith("gripper0_"):
            continue
        qa_o, va_o, jt = old_idx[name]
        if jt != new.jnt_type[j]:
            continue
        nq, nv = sizes[mujoco.mjtJoint(jt)]
        data.qpos[new.jnt_qposadr[j] : new.jnt_qposadr[j] + nq] = qpos_old[
            qa_o : qa_o + nq
        ]
        data.qvel[new.jnt_dofadr[j] : new.jnt_dofadr[j] + nv] = qvel_old[
            va_o : va_o + nv
        ]
        moved += 1
    raw_env.sim.forward()
    return moved


def _migrate_shell_inertia(root) -> int:
    """Move the legacy per-geom `shellinertia` flag onto the mesh asset.

    robocasa v1.0 scenes write `<geom type="mesh" shellinertia="true">`. Newer
    MuJoCo moved that switch to the asset and refuses the geom form with
    "for mesh geoms, inertia should be specified in the mesh asset".
    """
    meshes = {m.get("name"): m for m in root.iter("mesh")}
    migrated = 0
    for geom in root.iter("geom"):
        flag = geom.attrib.pop("shellinertia", None)
        if flag not in ("true", "1"):
            continue
        mesh = meshes.get(geom.get("mesh"))
        if mesh is not None:
            mesh.set("inertia", "shell")
        migrated += 1
    return migrated


def _repair_xml_assets(xml: str) -> str:
    """Cheap, idempotent fixes needed before any MuJoCo compile.

    Returns the input unchanged when there is nothing to fix, so datasets that
    already load cleanly compile exactly the same XML as before.
    """
    root = ET.fromstring(xml)
    changed = (
        _migrate_shell_inertia(root)
        + _relocate_asset_paths(root)
        + _strip_multiccd(root)
    )
    return ET.tostring(root, encoding="unicode") if changed else xml


KEEP_MULTICCD = False
_MULTICCD_NOTICED = False


def _strip_multiccd(root) -> int:
    """Turn MuJoCo's multi-point convex collision off.

    The Robotiq linkages self-collide slightly by construction, and with
    multiccd the extra contact points per pair lock the four-bar: the gripper
    stalls half open. Measured on robocasa v1.0 scenes, which bake
    `<flag multiccd="enable"/>` into every episode: 81mm -> 69mm instead of
    81mm -> 29mm, with the inner finger joint driven to -0.634 rad instead of
    -0.14. Disabling it reproduces the BARX behaviour (28.9mm) exactly, and
    BARX -- which every earlier cross-embodiment result came from -- never
    enabled the flag. Pass --keep-multiccd to leave the recorded setting alone.
    """
    global _MULTICCD_NOTICED
    if KEEP_MULTICCD:
        return 0
    changed = 0
    for opt in root.iter("option"):
        for flag in opt.findall("flag"):
            if flag.get("multiccd") not in (None, "disable"):
                flag.set("multiccd", "disable")
                changed += 1
    if changed and not _MULTICCD_NOTICED:
        print("[eval] disabled multiccd (it locks the Robotiq linkage half-open)")
        _MULTICCD_NOTICED = True
    return changed


_ASSET_ROOT_CACHE: list | None = None


def _asset_roots() -> list:
    """Local directories an episode's baked asset paths can be re-rooted onto.

    Episode XMLs carry absolute paths from the machine that recorded them
    (`/root/robocasa/...`). On top of that, object sets differ between checkouts
    -- the lightwheel objects used by robocasa v1.0 datasets ship in the
    standalone robocasa tree but not in BARX's robocasa_x -- so more than one
    root has to be searched. Extra roots can be added with
    ROBOCASA_EXTRA_ASSET_ROOTS (colon-separated).
    """
    global _ASSET_ROOT_CACHE
    if _ASSET_ROOT_CACHE is not None:
        return _ASSET_ROOT_CACHE
    roots = []
    for name in ("robocasa", "robosuite"):
        try:
            mod = __import__(name)
            roots.append(Path(mod.__file__).parent / "models" / "assets")
        except Exception:
            pass
    for extra in os.environ.get("ROBOCASA_EXTRA_ASSET_ROOTS", "").split(":"):
        if extra:
            roots.append(Path(extra))
    roots.append(Path("/data1/local/robocasa/robocasa/models/assets"))
    _ASSET_ROOT_CACHE = [r for r in dict.fromkeys(roots) if r.is_dir()]
    return _ASSET_ROOT_CACHE


def _relocate_asset_paths(root) -> int:
    """Re-root absolute asset paths that do not resolve on this machine."""
    marker = "models/assets/"
    relocated = 0
    for elem in root.iter():
        path = elem.get("file")
        if not path or not path.startswith("/") or os.path.exists(path):
            continue
        idx = path.find(marker)
        if idx < 0:
            continue
        suffix = path[idx + len(marker) :]
        for base in _asset_roots():
            candidate = base / suffix
            if candidate.exists():
                elem.set("file", str(candidate))
                relocated += 1
                break
    return relocated


def _sanitize_episode_xml(xml: str) -> str:
    root = ET.fromstring(xml)
    _migrate_shell_inertia(root)
    _relocate_asset_paths(root)
    used_meshes = {geom.get("mesh") for geom in root.iter("geom") if geom.get("mesh")}
    _remove_mesh_assets(root, lambda mesh_name: mesh_name not in used_meshes)

    # Some BARX visual meshes are referenced but fail MuJoCo's inertial mesh
    # compilation under the eval venv. Drop only the offending visual geom/mesh.
    try:
        import mujoco
    except Exception:
        return ET.tostring(root, encoding="unicode")

    def drop(mesh_name: str) -> int:
        removed = _remove_mesh_assets(root, lambda n: n == mesh_name)
        return removed + _remove_mesh_geoms(root, mesh_name)

    shelled = None  # mesh we last tried to rescue with a shell inertia
    for _ in range(60):
        candidate = ET.tostring(root, encoding="unicode")
        try:
            mujoco.MjModel.from_xml_string(candidate)
            return candidate
        except ValueError as exc:
            match = re.search(r"mesh volume is too small: ([^ .]+)", str(exc))
            if match is None:
                # The shell rescue upset something else -- undo it and fall back
                # to dropping the mesh, which is the behaviour without it.
                if shelled is None:
                    raise
                for mesh in root.iter("mesh"):
                    if mesh.get("name") == shelled:
                        mesh.attrib.pop("inertia", None)
                if drop(shelled) == 0:
                    raise
                shelled = None
                continue
            bad_mesh = match.group(1)
            # Prefer what MuJoCo itself suggests: a shell inertia keeps the mesh
            # (and whatever object it draws) instead of deleting it.
            if bad_mesh != shelled and any(
                m.get("name") == bad_mesh and m.get("inertia") != "shell"
                for m in root.iter("mesh")
            ):
                for mesh in root.iter("mesh"):
                    if mesh.get("name") == bad_mesh:
                        mesh.set("inertia", "shell")
                shelled = bad_mesh
                continue
            if drop(bad_mesh) == 0:
                raise
            shelled = None
    return ET.tostring(root, encoding="unicode")


def _install_fixture_state_guard() -> None:
    """Let per-step fixture updates skip fixtures this episode's scene lacks.

    The env's Python `Fixture` objects come from the scene sampled when the env
    was constructed. robocasa v1.0 datasets draw each episode from a different
    layout, so an episode legitimately has no `sink_right_group` -- and
    `Kitchen.update_state` then dies looking up `sink_right_group_handle_joint`,
    killing the run at the first such episode. Skip those fixtures instead;
    `_refresh_fixture_poses` is what keeps the ones that do exist pointing at
    the right place.
    """
    import robocasa.environments.kitchen.kitchen as _K

    if getattr(_K.Kitchen, "_barx_state_guard", False):
        return
    original = _K.Kitchen.update_state
    skipped: set[str] = set()

    def update_state(self):
        # Kitchen.update_state loops the fixtures itself, so a single failure
        # would skip every fixture after it; drive the loop here instead.
        super(_K.Kitchen, self).update_state()
        for name, fixtr in self.fixtures.items():
            try:
                fixtr.update_state(self)
            except (ValueError, KeyError) as exc:
                if name not in skipped:
                    skipped.add(name)
                    print(
                        f"[eval] fixture '{name}' is absent from this episode's scene, "
                        f"skipping its state update ({type(exc).__name__})"
                    )

    _K.Kitchen.update_state = update_state
    _K.Kitchen._barx_state_guard = True


def _install_object_reference_guard() -> None:
    """Let object-id lookup skip objects this episode's scene lacks.

    `Kitchen._setup_references` loops `self.objects` (the Python object list
    from whatever random scene `_make_env`'s *first* construction sampled)
    and hard-crashes via `body_name2id` the moment one is missing. Tasks like
    TurnOnSinkFaucet sample a variable number of distractors, so the initial
    construction can end up with e.g. `distr_counter_0` AND `distr_counter_1`
    while a specific stored episode -- loaded later via
    `reset_from_xml_string`, which reruns this same `_setup_references` --
    only ever had one distractor. That one extra name is enough to kill the
    whole run at `_reset_to_episode`, nondeterministically (it depends on how
    many distractors the *unrelated* first-construction scene happened to
    sample), the same class of bug `_install_fixture_state_guard` already
    survives for fixtures.
    """
    import robocasa.environments.kitchen.kitchen as _K

    if getattr(_K.Kitchen, "_barx_object_ref_guard", False):
        return
    skipped: set[str] = set()

    def _setup_references(self):
        # Kitchen._setup_references' own body is what crashes (the
        # `self.obj_body_id[name] = body_name2id(...)` loop below) -- calling
        # it first would just hit the same crash before this guard ever runs.
        # Skip straight to Kitchen's parent (the generic robosuite reference
        # setup: robot/gripper site ids etc., unrelated to self.objects) and
        # reimplement Kitchen's own per-object loop here with a skip guard.
        super(_K.Kitchen, self)._setup_references()
        self.obj_body_id = {}
        live = set(self.sim.model.body_names)
        for name, model in self.objects.items():
            if model.root_body not in live:
                if name not in skipped:
                    skipped.add(name)
                    print(
                        f"[eval] object '{name}' is absent from this episode's scene, "
                        f"skipping its body-id reference (KeyError-safe)"
                    )
                continue
            self.obj_body_id[name] = self.sim.model.body_name2id(model.root_body)

    _K.Kitchen._setup_references = _setup_references
    _K.Kitchen._barx_object_ref_guard = True


def _install_sink_handle_fallback() -> None:
    """Let sink handle read/write survive a stale sink identity.

    Same root cause as `_install_object_reference_guard`: `self.sink` is
    resolved once against whatever random scene `_make_env`'s first
    construction sampled, and never re-resolved for the episode that actually
    gets loaded via `reset_from_xml_string`. `Sink.set_handle_state` and
    `Sink.get_handle_state` hardcode `f"{self.naming_prefix}handle_joint"`
    with no fallback, so an episode whose layout has `sink_island_group`
    while the stale `self.sink` says `sink_left_group` dies immediately in
    `_reset_internal` (which calls `set_handle_state` to force a deterministic
    starting position -- a value that gets overwritten moments later anyway
    by the recorded state restore, so skipping it here is free). There is
    normally exactly one sink per kitchen, so temporarily swapping in
    whatever live joint actually matches `*_handle_joint`/`*_spout_joint`
    finds the right one regardless of which instance name this episode's
    layout gave it, and lets the unmodified original methods do the rest.
    """
    import robocasa.models.fixtures.sink as _S

    if getattr(_S.Sink, "_barx_handle_fallback", False):
        return

    def _resolve_name(self, env) -> str:
        # naming_prefix is a read-only property (f"{self.name}_"); swap the
        # underlying `_name` it's derived from instead.
        if f"{self.naming_prefix}handle_joint" in env.sim.model.joint_names:
            return self._name
        matches = [
            n
            for n in env.sim.model.joint_names
            if n.endswith("handle_joint") and "sink" in n
        ]
        return matches[0][: -len("_handle_joint")] if len(matches) == 1 else self._name

    def _with_resolved_prefix(method):
        def wrapper(self, env, *args, **kwargs):
            name = self._name
            handle_joint_cache = self._handle_joint
            self._name = _resolve_name(self, env)
            # `handle_joint` (the `_handle_joint is None` guard `get_handle_state`
            # checks first) searches `self.worldbody` -- this fixture instance's
            # own static XML, permanently baked with the *original* names --
            # for `{naming_prefix}handle_joint`, so it can never find a match
            # under a swapped name no matter what `self._name` is; short it to
            # a truthy sentinel so the guard reads "yes, this fixture type has
            # a handle joint" without redoing that dead-end search.
            self._handle_joint = True
            try:
                return method(self, env, *args, **kwargs)
            except ValueError:
                return {} if method.__name__ == "get_handle_state" else None
            finally:
                self._name = name
                self._handle_joint = handle_joint_cache

        return wrapper

    _S.Sink.set_handle_state = _with_resolved_prefix(_S.Sink.set_handle_state)
    _S.Sink.get_handle_state = _with_resolved_prefix(_S.Sink.get_handle_state)
    _S.Sink.update_state = _with_resolved_prefix(_S.Sink.update_state)
    _S.Sink._barx_handle_fallback = True


def _refresh_fixture_poses(raw_env, xml: str) -> int:
    """Re-point the Python fixture objects at the episode's actual scene.

    `reset_from_xml_string` swaps the MuJoCo model but leaves the env's Python
    `Fixture` objects untouched, and those still describe whatever random scene
    was built when the env was constructed. Success checks read fixture
    geometry through them (`fixture.pos`/`fixture.rot` -> `get_int_sites`), so
    without this every episode is scored against the *first* episode's sink /
    counter position. Measured effect: ground-truth demos that visibly place
    the object in the sink were being marked failures, because the sink was
    taken to be ~30-60cm away from where it actually is in that layout.

    Only pose is refreshed -- the fixtures' own interior-bound offsets are
    intrinsic to the fixture model and stay valid.
    """
    root = ET.fromstring(xml)
    bodies = {b.get("name"): b for b in root.iter("body") if b.get("name")}
    # Group by (leading family token, trailing role token) -- e.g. "sink"/"main"
    # -- not just the leading token. A single sink fixture already contributes
    # 3 same-family bodies (`sink_right_group_main`, `..._spout`, `..._handle`),
    # so grouping on "sink" alone always finds >=3 "candidates" and the old
    # ambiguity guard skipped the refresh for EVERY renamed sink, not just
    # genuinely ambiguous ones (multiple distinct counters). Keeping the role
    # token disambiguates `sink_left_group_main` -> `sink_right_group_main`
    # while still refusing to guess between two unrelated `_main` bodies of
    # the same family (e.g. two separate counters).
    by_family: dict[tuple[str, str], list] = {}
    for name in bodies:
        parts = name.split("_")
        by_family.setdefault((parts[0], parts[-1]), []).append(name)
    refreshed = 0
    for _name, fixture in getattr(raw_env, "fixtures", {}).items():
        element = getattr(fixture, "_obj", None)
        if element is None:
            continue
        body = bodies.get(element.get("name"))
        if body is None:
            # v1.0 datasets draw each episode from a different layout, so the
            # same sink can be `sink_right_group` in one and `sink_island_group`
            # in another. Fall back to the family when it names exactly one
            # body, and leave genuinely ambiguous families (several counters)
            # alone rather than guess.
            target_parts = (element.get("name") or "").split("_")
            family = (target_parts[0], target_parts[-1]) if target_parts[0] else None
            candidates = by_family.get(family, []) if family else []
            if len(candidates) != 1:
                continue
            body = bodies[candidates[0]]
        if body.get("pos"):
            element.set("pos", body.get("pos"))
        if body.get("euler"):
            element.set("euler", body.get("euler"))
        elif body.get("quat"):
            element.attrib.pop("euler", None)
            element.set("quat", body.get("quat"))
        refreshed += 1
    return refreshed


def _install_containment_tolerance(tol: float) -> None:
    """Give the `partial_check` containment test the same slack the full check has.

    BARX's task variants call `obj_inside_of(..., partial_check=True)`, which
    tests only the object's center against the fixture's interior box with zero
    tolerance, whereas the 8-corner path allows 5cm. A small object resting on
    the sink floor therefore has its *center* a few cm below the box's lower z
    bound and is judged outside -- ground-truth demos with the object plainly
    sitting in the sink were failing this way.
    """
    import numpy as _np
    import robocasa.utils.object_utils as _OU
    import robosuite.utils.transform_utils as _T

    if getattr(_OU, "_barx_tol_installed", False):
        return
    original = _OU.obj_inside_of

    def obj_inside_of(env, obj_name, fixture_id, partial_check=False):
        if not partial_check or tol <= 0.0:
            return original(env, obj_name, fixture_id, partial_check=partial_check)
        fixture = env.get_fixture(fixture_id)
        p0, px, py, pz = fixture.get_int_sites(relative=False)
        point = _np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        for axis, far in ((px - p0, px), (py - p0, py), (pz - p0, pz)):
            lo, hi = _np.dot(axis, p0), _np.dot(axis, far)
            if not (min(lo, hi) - tol <= _np.dot(axis, point) <= max(lo, hi) + tol):
                return False
        return True

    _OU.obj_inside_of = obj_inside_of
    _OU._barx_tol_installed = True


def _install_fixture_contact_fallback() -> None:
    """Make fixture contact tests survive a stale fixture *identity*.

    `_refresh_fixture_poses` repairs where a fixture is, but not which one it
    is. The env's Python fixtures come from the scene built at construction
    time, and a different layout can number its fixtures differently -- e.g.
    the task holds `counter_1_left_group` while the episode's real counter is
    `counter_left_group`. Every geom name the stale fixture reports is then
    absent from the sim, so `check_contact` answers False no matter what is
    actually touching: PnPSinkToCounter could essentially never be scored a
    success, even with the container sitting squarely on the counter.

    When a fixture's geoms are all missing, fall back to the geoms that *are*
    in the sim for the same fixture family (same leading type token and same
    trailing group), which is the same physical fixture under the episode's
    own numbering.
    """
    import re as _re
    import robosuite.utils.sim_utils as _SU

    if getattr(_SU, "_barx_contact_fallback", False):
        return
    original = _SU.check_contact

    def _family(name: str):
        parts = name.split("_")
        head = parts[0]
        tail = "_".join(p for p in parts[1:] if not p.isdigit())
        return head, tail

    def check_contact(sim, geoms_1, geoms_2=None):
        def resolve(g):
            if g is None or isinstance(g, str) or not hasattr(g, "contact_geoms"):
                return g
            live = set(sim.model.geom_names)
            declared = g.contact_geoms
            named = [n for n in declared if n in live]
            # A *partial* match is exactly the stale-identity case this
            # function exists to survive, not proof the fixture resolved
            # correctly: a counter's `_base_*` side geoms are named the same
            # across every layout, but its `_top_*` surface geoms are
            # subdivided by that instance's actual footprint and only
            # coincidentally share a name across two different layouts. A
            # fixture that matches on 4 of its 14 geoms (all `_base_*`, none
            # of the `_top_*` an object would actually rest on) looks
            # "resolved" by a non-empty check, but `check_contact` against
            # just the base can never see an object sitting on top --
            # measured: a ground-truth successful demo's own recorded final
            # state scored `recep_on_counter=False` against its own stale
            # `self.counter`. Require every declared geom to be live before
            # trusting the fixture as-is.
            if declared and len(named) == len(declared):
                return g
            prefix = getattr(g, "naming_prefix", None) or (getattr(g, "name", "") + "_")
            head, tail = _family(prefix.rstrip("_"))
            alt = [
                n
                for n in live
                if n.startswith(head + "_")
                and _family(n)[0] == head
                and tail
                and tail.split("_")[-1] in n
            ]
            return alt or named or g

        return original(sim=sim, geoms_1=resolve(geoms_1), geoms_2=resolve(geoms_2))

    _SU.check_contact = check_contact
    _SU._barx_contact_fallback = True


def _remove_mesh_assets(root, predicate) -> int:
    removed = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "mesh" and predicate(child.get("name")):
                parent.remove(child)
                removed += 1
    return removed


def _remove_mesh_geoms(root, mesh_name: str) -> int:
    removed = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "geom" and child.get("mesh") == mesh_name:
                parent.remove(child)
                removed += 1
    return removed


def _filter_model_geom_refs_to_xml(model, xml: str) -> None:
    root = ET.fromstring(xml)
    xml_geoms = {geom.get("name") for geom in root.iter("geom") if geom.get("name")}
    xml_sites = {site.get("name") for site in root.iter("site") if site.get("name")}

    def filter_one(m):
        for attr, valid in (
            ("visual_geoms", xml_geoms),
            ("contact_geoms", xml_geoms),
            ("sites", xml_sites),
        ):
            names = getattr(m, attr, None)
            if names is not None:
                filtered = [name for name in names if name in valid]
                try:
                    setattr(m, attr, filtered)
                except AttributeError:
                    if hasattr(names, "__setitem__"):
                        names[:] = filtered

    filter_one(model)
    for obj in getattr(model, "mujoco_objects", []) or []:
        filter_one(obj)
    for robot in getattr(model, "mujoco_robots", []) or []:
        filter_one(robot)
        for submodel in getattr(robot, "models", []) or []:
            filter_one(submodel)


def _align_robot_gripper_joints_to_xml(raw_env, xml: str) -> None:
    root = ET.fromstring(xml)
    xml_joints = {
        joint.get("name") for joint in root.iter("joint") if joint.get("name")
    }
    xml_actuators = {
        actuator.get("name")
        for tag in ("motor", "position", "velocity", "general")
        for actuator in root.iter(tag)
        if actuator.get("name")
    }
    for robot in getattr(raw_env, "robots", []):
        gripper_joints = getattr(robot, "gripper_joints", None)
        if not isinstance(gripper_joints, dict):
            continue
        for arm, names in list(gripper_joints.items()):
            gripper = getattr(robot, "gripper", {}).get(arm)
            if any(name in xml_joints for name in names):
                gripper_joints[arm] = [name for name in names if name in xml_joints]
                replacement = gripper_joints[arm]
            else:
                prefix = f"gripper0_{arm}_"
                replacement = sorted(
                    name for name in xml_joints if name.startswith(prefix)
                )
            if replacement:
                gripper_joints[arm] = replacement
                if gripper is not None and hasattr(gripper, "_joints"):
                    prefix = f"gripper0_{arm}_"
                    gripper._joints = [
                        name.removeprefix(prefix) for name in replacement
                    ]
            if gripper is not None and hasattr(gripper, "_actuators"):
                prefix = f"gripper0_{arm}_"
                actuators = sorted(
                    name for name in xml_actuators if name.startswith(prefix)
                )
                if actuators:
                    gripper._actuators = [
                        name.removeprefix(prefix) for name in actuators
                    ]
                    gripper.current_action = np.zeros(len(actuators))
                    if (
                        gripper.__class__.__name__ == "Robotiq85Gripper"
                        and len(actuators) > gripper.dof
                    ):
                        _patch_robotiq85_xml_actuator_action(gripper, len(actuators))


def _patch_robotiq85_xml_actuator_action(gripper, actuator_dof: int) -> None:
    import types

    def format_action(self, action):
        action = np.asarray(action).reshape(-1)
        assert len(action) == 1
        current = np.asarray(
            getattr(self, "_xml_actuator_current_action", np.zeros(actuator_dof)),
            dtype=float,
        )
        current = np.clip(
            current + np.ones(actuator_dof) * self.speed * np.sign(action[0]), -1.0, 1.0
        )
        self._xml_actuator_current_action = current
        self.current_action = current
        return current

    gripper._xml_actuator_current_action = np.zeros(actuator_dof)
    gripper.format_action = types.MethodType(format_action, gripper)


def _state_from_obs(obs: dict, modality: dict) -> np.ndarray:
    state_info = modality["state"]
    state_dim = max(item["end"] for item in state_info.values())
    state = np.zeros(state_dim, dtype=np.float32)
    for key, info in state_info.items():
        obs_key = LEROBOT_STATE_TO_OBS_KEY[key]
        value = np.asarray(obs[obs_key], dtype=np.float32).reshape(-1)
        width = info["end"] - info["start"]
        if key == "gripper_qpos" and value.shape[0] != width:
            value = np.asarray([value[0], value[-1]], dtype=np.float32)
        state[info["start"] : info["end"]] = value
    return state


def _policy_observation(obs: dict, modality: dict, input_features: dict) -> dict:
    policy_obs = {
        "observation.state": np.ascontiguousarray(_state_from_obs(obs, modality))
    }
    for key, feature in input_features.items():
        feature_type = getattr(feature.type, "name", str(feature.type))
        if feature_type != "VISUAL":
            continue
        camera_name = key.removeprefix("observation.images.")
        obs_key = f"{camera_name}_image"
        if obs_key not in obs:
            raise KeyError(f"Camera observation {obs_key!r} missing from env obs")
        policy_obs[key] = np.ascontiguousarray(obs[obs_key])
    return policy_obs


def _lerobot_action_to_env_action(
    action: np.ndarray, modality: dict, env_action_dim: int
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    action_info = modality["action"]
    raw_dim = max(end for _, end in ACTION_KEY_ORDERING_HDF5.values())
    raw_action = np.zeros(raw_dim, dtype=np.float64)
    for key, (raw_start, raw_end) in ACTION_KEY_ORDERING_HDF5.items():
        if key not in action_info:
            continue
        le_start = action_info[key]["start"]
        le_end = action_info[key]["end"]
        raw_action[raw_start:raw_end] = action[le_start:le_end]
    return raw_action[:env_action_dim]


def _render_frame(env, camera_names: list[str], height: int, width: int) -> np.ndarray:
    frames = [
        env.render(
            mode="rgb_array", height=height, width=width, camera_name=camera_name
        )
        for camera_name in camera_names
    ]
    return np.concatenate(frames, axis=1)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def evaluate(args: argparse.Namespace) -> dict:
    _add_local_python_paths(args.lerobot_src, args.robosuite_src, args.robocasa_src)
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not args.random_actions and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    env_meta, modality = _load_dataset_metadata(dataset)
    if args.gripper == "Robotiq85GripperLegacy":
        _register_legacy_robotiq85()
    env_meta = _override_robot_setup(env_meta, args.robot, args.gripper)
    if args.random_actions:
        # No checkpoint to derive a default camera set from in this mode.
        cfg = _RandomPolicyConfig(args.video_cameras or ["robot0_agentview_right"])
        policy = preprocessor = postprocessor = None
        rng = np.random.default_rng(args.seed)
    else:
        cfg, policy, preprocessor, postprocessor = _load_lerobot_policy(
            checkpoint, args.device
        )
        rng = None
    if args.robocasa_x_compat:
        _install_robocasa_x_compat_shim()
    if args.success_inside_tol > 0:
        _install_containment_tolerance(args.success_inside_tol)
    global KEEP_MULTICCD
    KEEP_MULTICCD = args.keep_multiccd
    _install_fixture_contact_fallback()
    _install_fixture_state_guard()
    _install_object_reference_guard()
    _install_sink_handle_fallback()
    _install_mesh_sanitizer()
    _policy_cam_h, _policy_cam_w = _camera_resolution_from_features(cfg.input_features)
    _policy_cameras = _camera_names_from_features(cfg.input_features)
    print(
        f"[eval] rendering policy observations at {_policy_cam_h}x{_policy_cam_w} "
        f"(from the checkpoint's own input_features)"
    )
    env = _make_env(
        env_meta,
        args.seed,
        _policy_cameras,
        camera_height=_policy_cam_h,
        camera_width=_policy_cam_w,
    )
    # Video defaults to exactly what the policy saw -- same camera set, same
    # resolution/aspect ratio (not just "also 180x320" but the identical
    # buffer shape, so this is literally a recording of the policy's input,
    # not a same-size-but-differently-cropped view of the same camera).
    if args.video_cameras is None:
        args.video_cameras = _policy_cameras
    if args.video_height is None:
        args.video_height = _policy_cam_h
    if args.video_width is None:
        args.video_width = _policy_cam_w
    print(
        f"[eval] saving video from {args.video_cameras} at "
        f"{args.video_height}x{args.video_width} (matches policy input exactly)"
    )
    # arm transplant: needed when --robot has no dataset of its own
    dataset_robot = env_meta.get("env_kwargs", {}).get("robots") or [None]
    dataset_robot = (
        dataset_robot[0] if isinstance(dataset_robot, (list, tuple)) else dataset_robot
    )
    try:
        robot_donor_xml = env.env.model.get_xml()
        env.env._robot_donor_xml = robot_donor_xml
        env._robot_donor_xml = robot_donor_xml
    except Exception as exc:
        print(f"[warn] could not capture robot donor XML: {exc}")

    donor_xml = None
    if args.gripper_donor_dataset is not None:
        # copy the gripper verbatim from another dataset's recorded scene
        donor_states, _, donor_xml = _episode_payload(args.gripper_donor_dataset, 0)
        if donor_xml is not None:
            donor_qpos = _donor_gripper_qpos(
                env.env.edit_model_xml(donor_xml), donor_states[0]
            )
            env._gripper_donor_qpos = donor_qpos
            env.env._gripper_donor_qpos = donor_qpos
            donor_xml = env.env.edit_model_xml(donor_xml)
            print(
                f"[eval] gripper donor: {args.gripper_donor_dataset} "
                f"({len(donor_qpos)} gripper joints seeded)"
            )
    elif args.gripper and args.gripper != "dataset":
        # `env.model` is the Python world robosuite built from --robot/--gripper;
        # reset_from_xml_string never touches it, so it stays a clean donor for
        # transplanting the requested gripper into each episode's stored scene.
        try:
            donor_xml = env.env.model.get_xml()
        except Exception as exc:
            print(f"[warn] could not capture gripper donor XML: {exc}")
    if donor_xml is not None:
        env.env._gripper_donor_xml = donor_xml
        env._gripper_donor_xml = donor_xml
    paths = _episode_paths(dataset, args.episodes, args.num_episodes)
    fixed_camera_configs = None
    if args.fixed_camera_episode is not None:
        _, fixed_camera_meta, _ = _episode_payload(dataset, args.fixed_camera_episode)
        recorded_configs = fixed_camera_meta.get("cam_configs", {})
        fixed_camera_configs = {
            name: deepcopy(recorded_configs[name])
            for name in _policy_cameras
            if name in recorded_configs
        }
        missing = sorted(set(_policy_cameras) - set(fixed_camera_configs))
        if missing:
            raise KeyError(
                f"fixed camera episode {args.fixed_camera_episode} has no configs for {missing}"
            )
        print(
            f"[eval] fixing {sorted(fixed_camera_configs)} to camera pose(s) from "
            f"episode {args.fixed_camera_episode}"
        )

    results = []
    success_count = 0
    with torch.inference_mode():
        from lerobot.policies.utils import prepare_observation_for_inference

        for rollout_i, parquet_path in enumerate(
            tqdm(paths, desc="Evaluating episodes")
        ):
            ep_idx = _episode_index(parquet_path)
            episode_df = pd.read_parquet(parquet_path)
            states, ep_meta, model_xml = _episode_payload(dataset, ep_idx)
            if fixed_camera_configs is not None:
                ep_meta = deepcopy(ep_meta)
                ep_meta.setdefault("cam_configs", {}).update(
                    deepcopy(fixed_camera_configs)
                )
            obs = _reset_to_episode(env, states[0], ep_meta, model_xml)
            if hasattr(policy, "reset"):
                policy.reset()

            base_steps = len(episode_df)
            max_steps = (
                args.max_steps
                if args.max_steps is not None
                else base_steps + args.extra_steps
            )
            save_video = args.save_videos and (
                args.num_video_episodes is None or rollout_i < args.num_video_episodes
            )
            frames = []
            success = False
            for step in range(max_steps):
                if save_video and step % args.video_skip == 0:
                    frames.append(
                        _render_frame(
                            env, args.video_cameras, args.video_height, args.video_width
                        )
                    )

                if args.random_actions:
                    env_action = _sample_random_env_action(
                        env, rng, args.random_action_scale
                    )
                else:
                    policy_obs = _policy_observation(obs, modality, cfg.input_features)
                    policy_obs = prepare_observation_for_inference(
                        policy_obs,
                        device=torch.device(args.device),
                        task=args.task,
                    )
                    policy_obs = preprocessor(policy_obs)
                    # Some LeRobot pre-processors keep only tensor observation keys and
                    # drop auxiliary strings. Diffusion checkpoints with
                    # language_conditioned=True require the task after preprocessing.
                    policy_obs["task"] = args.task
                    action = policy.select_action(policy_obs)
                    action = postprocessor(action)
                    env_action = _lerobot_action_to_env_action(
                        action.detach().cpu().numpy(),
                        modality=modality,
                        env_action_dim=env.action_dimension,
                    )
                obs, _, _, _ = env.step(env_action)
                success = bool(env.is_success().get("task", False))
                if success:
                    break

            if save_video and frames:
                suffix = f"success-{int(success)}"
                _write_video(
                    output_dir / "videos" / f"episode_{ep_idx:06d}_{suffix}.mp4",
                    frames,
                    fps=args.video_fps,
                )

            success_count += int(success)
            results.append(
                {
                    "episode": ep_idx,
                    "success": success,
                    "steps": step + 1,
                    "base_steps": base_steps,
                    "extra_steps": args.extra_steps if args.max_steps is None else 0,
                    "max_steps": max_steps,
                    "parquet": str(parquet_path),
                }
            )

    summary = {
        "checkpoint": None if args.random_actions else str(checkpoint),
        "random_actions": args.random_actions,
        "random_action_scale": args.random_action_scale
        if args.random_actions
        else None,
        "dataset": str(dataset),
        "task": args.task,
        "robot": args.robot or env_meta.get("env_kwargs", {}).get("robots"),
        "gripper": args.gripper or env_meta.get("env_kwargs", {}).get("gripper_types"),
        "num_episodes": len(results),
        "num_success": success_count,
        "success_rate": success_count / max(len(results), 1),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Wrote per-episode results to {output_dir / 'results.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lerobot-src", type=Path, default=DEFAULT_LEROBOT_SRC)
    parser.add_argument("--robosuite-src", type=Path, default=DEFAULT_ROBOSUITE_SRC)
    parser.add_argument("--robocasa-src", type=Path, default=DEFAULT_ROBOCASA_SRC)
    parser.add_argument(
        "--gripper-donor-dataset",
        type=Path,
        default=None,
        help=(
            "Take the gripper from this dataset's recorded episode XML instead of "
            "building it from --gripper. robosuite ships two different Robotiq 85 "
            "models (robotiq_gripper_85.xml and robotiq_2f85_menagerie) whose "
            "grip_site frames differ by 90deg, so 'Robotiq85Gripper' alone does not "
            "pin down which one a dataset used. Point this at the dataset you want "
            "to match and the gripper is copied verbatim."
        ),
    )
    parser.add_argument(
        "--success-inside-tol",
        type=float,
        default=0.05,
        help=(
            "Tolerance (m) for the center-point containment test used by BARX task "
            "variants. Matches the 5cm slack the 8-corner test already allows; 0 "
            "restores the original zero-tolerance behaviour."
        ),
    )
    parser.add_argument(
        "--keep-multiccd",
        action="store_true",
        help=(
            "keep the recorded multiccd setting. robocasa v1.0 scenes enable it, which "
            "locks a transplanted Robotiq gripper half open; off by default so the "
            "gripper matches the BARX runs."
        ),
    )
    parser.add_argument(
        "--robocasa-x-compat",
        action="store_true",
        help="load BARX robocasa_x in the LeRobot eval venv despite its strict version guards",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="run random env actions instead of loading a LeRobot checkpoint",
    )
    parser.add_argument(
        "--random-action-scale",
        type=float,
        default=1.0,
        help="scale random actions around the middle of env.action_spec",
    )
    parser.add_argument(
        "--robot", default=None, help="override env_kwargs.robots, e.g. UR5eOmron"
    )
    parser.add_argument(
        "--gripper",
        default=None,
        help="override env_kwargs.gripper_types; use 'dataset' to keep dataset metadata",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--fixed-camera-episode",
        type=int,
        default=None,
        help=(
            "Use the policy camera pose and FOV recorded in this episode for every "
            "rollout. Scene, objects, and initial state still come from each episode."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--extra-steps", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--num-video-episodes", type=int, default=5)
    parser.add_argument(
        "--video-cameras",
        nargs="+",
        default=None,
        help="Cameras to save video for. Default: exactly the cameras the checkpoint's "
        "own input_features use -- i.e. the video shows what the policy actually saw, "
        "not an arbitrary extra view it never had access to.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=None,
        help="Default: the checkpoint's own observation width. MuJoCo's camera fovy is a "
        "*vertical* FOV, so rendering at a different aspect ratio than the policy's "
        "actual input (e.g. a square 512x512 buffer for a 180x320 input) changes how "
        "much of the scene is visible -- the saved video would show a narrower crop "
        "than what the policy was actually fed, not just a lower-res version of it.",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=None,
        help="Default: the checkpoint's own observation height. See --video-width.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
