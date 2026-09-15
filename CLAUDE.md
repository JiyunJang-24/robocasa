# CLAUDE.md — cross-embodiment dataset generation and evaluation

This fork of [robocasa/robocasa](https://github.com/robocasa/robocasa) adds a
pipeline for **cross-embodiment** manipulation data: take a demonstration
recorded on one robot, and re-render the same task, the same trajectory, on
several other robot embodiments — so a policy (or an analysis) can be given
"the same motion, different arm" pairs it would otherwise never see.

Two things live on top of vanilla RoboCasa:

1. **Dataset generation** (`scripts/datagen/`) — replays a recorded episode's
   raw actions across multiple robots in the *same* scene, either with the
   task objects still present (a same-scene reproduction check) or flown out
   of frame (the actual cross-embodiment training data).
2. **Evaluation** (`scripts/eval/`) — loads a trained LeRobot policy
   checkpoint and rolls it out in a live RoboCasa/BARX environment. There are
   **two distinct eval setups**, because two different recording pipelines
   produced the checkpoints this repo evaluates — see "Two eval pathways"
   below. Getting the wrong one for a given checkpoint does not error; it
   silently evaluates the policy in a scene distribution it was never trained
   for.

Read this file before running anything. Several sections exist because a
default that *looks* safe produced a plausible-looking but wrong dataset —
see "Gotchas", all of them earned the hard way.

---

## 1. Repository layout

```
robocasa/                  the RoboCasa package itself (upstream, mostly unmodified)
mimicgen/                  git submodule -> robocasa-benchmark/mimicgen
robomimic/                 git submodule -> ARISE-Initiative/robomimic
third_party/
  barx/                    git submodule -> ajaysridhar0/barx (robocasa_x fork + atomic task defs)
  robosuite/                git submodule -> JiyunJang-24/robosuite @ robocasa-x-omron
                            (adds Omron composite-robot controller configs and
                            a mujoco_menagerie Robotiq 2F-85 model; see §4)
scripts/
  datagen/                 dataset generation & post-processing (§2)
  eval/                    policy checkpoint evaluation (§3)
  diagnostics/             one-off debugging scripts for action tracking / grasp execution
  smoke_tests/             ad-hoc "does this robot+gripper combo load" scripts, not a test suite
custom_robots/             menagerie-derived robot ports not upstreamed to robosuite (e.g. YAM, PiPER)
```

`datasets/` and `outputs/` are gitignored (generated data, easily 100s of GB).
Nothing under them is required to reproduce the pipeline — only the scripts
that produce them.

---

## 2. Dataset generation

### 2a. Where the source demonstrations come from

Everything here starts from an already-recorded, single-embodiment LeRobot
dataset ("atomic" data): one robot, one task, real MimicGen-generated or
human demonstrations. Two families of atomic data exist and are **not
interchangeable**:

| family | built with | registers via |
|---|---|---|
| `datasets/v1.0/...` (this repo's own atomic runs) | this repo's own `robocasa` package | plain `import robocasa` |
| `third_party/barx/datasets/atomic_new/...` | BARX's `robocasa_x` fork | `third_party/barx/robocasa_x` on `sys.path`, BARX's own venv |

`robocasa_x`'s Kitchen/fixture/object registries are a fork of this repo's,
close enough to look similar and different enough that fixture names (e.g.
`counter_left_group` vs `counter_1_left_left_group`) and object instance IDs
do not line up. **Reconstructing a `robocasa_x`-recorded episode needs
`robocasa_x` itself**, not this repo's `robocasa` package. This is why
`generate_visual_robust_cross_embodiment_x.py` inserts
`third_party/barx/robocasa_x` onto `sys.path` and must be run under BARX's own
venv (`third_party/barx/.venv`, `uv sync` inside that submodule to build it —
pins `mujoco==3.1.1`, `numpy==1.23.5`, which `robocasa_x`'s own import guards
assert).

### 2b. The generators

| script | input | what it produces |
|---|---|---|
| `generate_mimicgen_ur5e_dataset.py` | a MimicGen task config | fresh UR5e demonstrations, with the UR5e-specific workspace/backoff fixes documented at the top of the file |
| `generate_visual_robust_cross_embodiment.py` | this repo's own atomic data | multi-embodiment replay, LIBERO-style empty-scene technique ported to RoboCasa |
| `generate_visual_robust_cross_embodiment_x.py` | **BARX's `atomic_new`** | the current, actively-used generator — see below |
| `generate_barx_cross_embodiment_dataset.py` | BARX task configs | wraps `generate_mimicgen_ur5e_dataset.py` for BARX's per-embodiment initial poses |
| `atomic_same_scene_crossembodiment_demo.py` | one atomic episode | same-scene, same-objects reproduction on a different robot — the fidelity check to run *before* trusting a full cross-embodiment render |
| `render_barx_normalized_hdf5.py` | atomic demos | BARX's normalized HDF5 layout (`obs/agentview_rgb`, `ee_states`, `gripper_states`, `aux_info` bboxes) |

`generate_visual_robust_cross_embodiment_x.py` is the one to reach for by
default. For each source episode it:

1. Replays the source robot's exact recorded action sequence to reconstruct
   the scene (nothing is re-sampled — the episode is rebuilt from its own
   `ep_meta`/model XML, so every embodiment sees the identical fixture
   layout).
2. Re-solves the same trajectory's end-effector path on each requested target
   robot via damped-least-squares IK, matching fingertip position (and
   optionally orientation) at every step.
3. Hides the task objects (or the whole kitchen, or nothing — `--hide-scope`,
   see Gotcha below) before rendering, since the point is the *robot*, not a
   scene none of the other embodiments actually manipulated.
4. Writes one LeRobot dataset with one row per timestep and one image column
   *per requested embodiment* — `observation.images.{Tag}.{camera}` — so a
   single frame index gives you the same instant on every robot at once.

```bash
cd third_party/barx && uv sync && cd ../..   # build BARX's venv once

third_party/barx/.venv/bin/python scripts/datagen/generate_visual_robust_cross_embodiment_x.py \
  --source-lerobot-dir third_party/barx/datasets/atomic_new/IIWAOmron/pretrain/PnPCounterToSink/lerobot \
  --episodes 0 1 2 \
  --robots PandaOmron UR5eOmron IIWAOmron "JacoOmron:JacoOmron:Robotiq85Gripper" \
  --camera-names robot0_agentview_right robot0_eye_in_hand \
  --camera-height 180 --camera-width 320 \
  --hide-scope all_objects \
  --output-dir outputs/datagen/IIWAOmron_PnPCounterToSink
```

`--robots` entries are either a bare robosuite robot class (gripper resolved
from `ATOMIC_NEW_GRIPPER` — what that robot's own `atomic_new` recording
actually used) or `Tag:RobotClass:Gripper` to render a robot under a gripper
other than its own recording, or to add a robot with no `atomic_new` recording
of its own at all (e.g. Jaco, which was never a source robot).

For a large run, shard `--episodes` across processes/GPUs and merge with
`scripts/datagen/merge_visual_robust_shards.py --shard-dirs ... --shard-starts ...`
— it assumes each shard's `--episodes` was a contiguous ascending block and
remaps local episode index `i` back to the global source index
`shard_start + i`. Passing a non-contiguous or reordered `--episodes` list
per shard will silently mislabel episodes on merge.

### 2c. Matching an already-published dataset

If you're extending or re-generating a dataset that's already been published
(e.g. under `new_barx/` on the Hub), diff your intended flags against a
decoded frame from the existing one before committing to a long run —
**every flag below has a default that produces a plausible-looking dataset
with the wrong content**, and none of the three mistakes below throw an
error:

```python
import imageio.v2 as iio
r = iio.get_reader("<existing dataset>/videos/chunk-000/<Tag>.<camera>/episode_000000.mp4")
r.get_data(0).shape   # (height, width, 3) -- compare against your --camera-height/--width
```

---

## 3. Evaluation — two pathways, do not mix them up

Both go through `scripts/eval/eval_lerobot_checkpoint.py`, a policy-agnostic
LeRobot-checkpoint evaluator (it resolves the policy class from the
checkpoint's own `config.json`, so it works for Diffusion Policy, SmolVLA,
etc. without modification). What differs between the two launchers is **which
environment package builds the RoboCasa scene**, because that has to match
whichever package the checkpoint's *training data* was recorded with:

| | `scripts/eval/run_robocasa_eval.sh` | `scripts/eval/run_robocasa_x_eval.sh` |
|---|---|---|
| training data | this repo's own atomic/`v1.0` runs | BARX's `atomic_new` |
| env package | this repo's `robocasa` | BARX's `robocasa_x` (`--robocasa-x-compat`) |
| task text | fixed per invocation | read from the dataset's own `meta/tasks.jsonl` |
| robot override | none (uses the dataset's own robot) | `--robot`/`--gripper`, for cross-embodiment eval on a robot the checkpoint never trained on |
| env var prefix | — | `run_robocasa_x_eval.sh [ROBOT] [GRIPPER] [TASK]` |

Using the plain launcher on a BARX-trained checkpoint (or vice versa) doesn't
fail loudly: robosuite/robocasa_x's fixture and object registries are close
enough that the env *builds*, and the policy runs against a scene
distribution shifted just enough to be a bad benchmark of it.

```bash
export LEROBOT_SRC=/path/to/a/lerobot/src        # not vendored here, see §5
export PYTHON=/path/to/that/lerobots/venv/bin/python

# plain robocasa
CHECKPOINT=outputs/train/.../pretrained_model \
DATASET=datasets/v1.0/target/atomic/PickPlaceSinkToCounter/lerobot \
  scripts/eval/run_robocasa_eval.sh

# BARX-style, cross-embodiment (checkpoint trained on UR5e, evaluated on Jaco)
CHECKPOINT=outputs/train/.../pretrained_model \
  scripts/eval/run_robocasa_x_eval.sh JacoOmron Robotiq85Gripper PnPSinkToCounter
```

`run_robocasa_x_eval.sh` pins `Robotiq85Gripper` to the mesh recorded in the
UR5eOmron `atomic_new` tree via `--gripper-donor-dataset`: robosuite's own
`Robotiq85Gripper` class resolves to a different model
(`2f85_robosuite.xml`, `base_mount` root, 13 bodies) than what every
`atomic_new` recording actually used
(`robotiq_2f85_menagerie/2f85_robosuite.xml`, `robotiq_85_adapter_link` root,
8 bodies) — see §4. Passing `--gripper Robotiq85Gripper` with no donor
dataset silently substitutes a gripper the policy has never seen the shape
of.

`scripts/eval/run_dp_embodiment_*.sh` are sweep launchers over the
language-conditioned Diffusion Policy checkpoints in
`outputs/train/dp_embodiment/{scratch,frozen,finetune,online}` — see the
docstrings in `run_dp_embodiment_barx_pnpsink_parallel.sh` for the
scratch/frozen/finetune/online distinction (which parts of the vision tower
are pre-trained vs. trainable).

---

## 4. The `third_party/robosuite` fork — what it adds and why

Vanilla robosuite has no Omron mobile-base composite controllers and ships a
Robotiq 2F-85 model that is *not* the one BARX's recordings used. Both are
required to render `atomic_new` cross-embodiment data, so this repo pins a
fork (`JiyunJang-24/robosuite`, branch `robocasa-x-omron`) rather than
upstream:

- `robosuite/controllers/config/robots/default_{iiwa,jaco,ur5e}omron.json` —
  composite controller configs for the Omron-base variants. Loading them
  through `load_composite_controller_config(controller=path)` and not a raw
  JSON parse matters: the raw JSON needs `body_parts.arms` flattened first or
  `robot.reset` raises `KeyError: 'right'`.
- `robosuite/models/assets/grippers/robotiq_2f85_menagerie/` — the
  mujoco_menagerie Robotiq 2F-85, matching what `atomic_new` recorded.
- Three small changes to `robotiq_85_gripper.py` / `compositional.py` /
  `robots/__init__.py` wiring the above in.

If robosuite drifts and this fork needs rebasing: the only real content is
those 18 files (`git log robocasa-x-omron` in the submodule), so a rebase is
re-applying a small, self-contained diff, not resolving a large merge.

---

## 5. Setup

```bash
git clone --recurse-submodules https://github.com/JiyunJang-24/robocasa.git
cd robocasa
# or, if already cloned without --recurse-submodules:
git submodule update --init --recursive
```

Two separate Python environments are needed, because `robocasa_x` pins
different core-package versions than this repo's own `robocasa`:

| env | for | build |
|---|---|---|
| `robocasa` (conda, see root README §Installation) | this repo's own dataset gen / plain eval | `conda create -n robocasa python=3.11`, `pip install -e third_party/robosuite && pip install -e .` |
| BARX's own venv | anything touching `atomic_new` / `robocasa_x` (§2a) | `cd third_party/barx && uv sync` |

**LeRobot is intentionally not vendored.** The scripts under `scripts/eval/`
take `--lerobot-src` / `LEROBOT_SRC` and `--checkpoint` as parameters
precisely so this repo doesn't need to track a specific LeRobot fork's
training code — point them at whatever produced the checkpoint you're
evaluating (a `pip install lerobot`, or a private training fork). If the eval
venv is missing `lxml`, `.eval_extra_site/` is a `sys.path` entry the launcher
scripts already include for exactly this — put a symlink to a `site-packages`
copy there rather than modifying the venv.

**Not every script under `scripts/datagen/` has had its default paths
audited.** `generate_visual_robust_cross_embodiment_x.py` and the two
`scripts/eval/run_robocasa*.sh` launchers are: their path defaults resolve
relative to the repo root (`ROBOCASA_ROOT`, `BARX_ROOT`, `ROBOSUITE_SRC` env
vars override them). Older one-off scripts may still have a development
machine's absolute paths as defaults — check the top-of-file constants
before running one you haven't used before.

---

## 6. Gotchas

Ranked by how expensive they were to find, since each one produced output
that looked fine until inspected directly.

**`--hide-scope` semantics are not what the names suggest.** `robot_only`
does *not* mean "hide only the robot's surroundings, keep the kitchen" — it
hides the kitchen fixtures *and* the task objects, leaving only the robot
against a bare void. `all_objects` is the one that keeps counters/sink/
cabinets visible and flies away just the manipulable objects, which is what
"cross-embodiment, same background" almost always wants. Both produce a
clean-looking render; the only way to catch the mix-up is to decode a frame
and look.

**Camera resolution silently defaults to 256x256 square.** Every `atomic_new`
recording and every previously-published cross-embodiment export is
180x320. If `--camera-height`/`--camera-width` are omitted, the generator's
own defaults (256, 256) apply with no warning, and the run completes
successfully at the wrong aspect ratio. Always pass both explicitly and spot
check `frame.shape` on the first output episode before launching a long run.

**Mesh `inertia` attribute needs a compat shim on MuJoCo < 3.1.4.**
`<mesh inertia="shell|legacy|exact|convex">` is rejected as a schema
violation on older MuJoCo. Several robots/grippers used here ship it
(Sawyer's own `robot.xml`, several `robosuite_models` grippers).
`install_mesh_inertia_compat()` strips the attribute at XML-parse time
(every affected body already declares an explicit `<inertial>`, so this
changes nothing about the physics) — call it before building any env.

**Gripper choice is not cosmetic.** `ATOMIC_NEW_GRIPPER` in
`generate_visual_robust_cross_embodiment_x.py` records which gripper each
robot's own `atomic_new` recording actually used (PandaOmron:
`Robotiq85Gripper`, not Panda's native `PandaGripper`; UR5eOmron/IIWAOmron:
each robot's own composite default). Rendering an alternate embodiment with
whatever gripper the robot class defaults to, instead of this table, silently
gives it a gripper — and thus a fingertip offset and action semantics — it
was never recorded with.

**`env.reset()` on a `robosuite.MjSim`-backed env invalidates cached
handles.** It frees and rebuilds the simulator, so any cached `sim`, site id,
or joint index from before the reset is dangling afterward — this surfaces
as `'MjSim' object has no attribute 'data'`, which reads like an API version
mismatch rather than what it is. Re-fetch handles after every reset, or avoid
resetting mid-loop (restore a saved qpos vector instead).

**PnP task names differ between robocasa and robocasa_x.** v1.0 renamed
`PnP*` to `PickPlace*`; MimicGen and `atomic_new`'s recorded `env_name` still
use the old names, and `robocasa_x` records an `X`-prefixed variant
(`XPnPCounterToSink`) on top of that. `resolve_env_name()` tries the
un-prefixed name, then both `PnP`/`PickPlace` spellings, against whichever
package's `REGISTERED_ENVS` is actually imported — needed because "not
registered" from a raw `robosuite.make(...)` call gives no hint that the fix
is a name, not an environment, problem.
