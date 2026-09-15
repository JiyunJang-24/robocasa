#!/bin/bash
# Render every generated success shard into the normalized BARX raw layout.
# Usage: GPU=<id> LANE=<panda|ur5e|iiwa|kinova3> bash render_all_barx_shards.sh
set -e
PY=/home/fick17/anaconda3/envs/robocasa/bin/python
CE=/data1/local/BARX/datasets/cross_embodiment
RAW=/data1/local/BARX/datasets/barx-raw
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${GPU:-0}
cd /data1/local/robocasa

render() {  # render <output-rel-path> <obj-names...> -- <inputs...>
  local out=$1; shift
  local objs=()
  while [ "$1" != "--" ]; do objs+=("$1"); shift; done
  shift
  $PY render_barx_normalized_hdf5.py --output "$RAW/$out" \
      --obj-names "${objs[@]}" --inputs "$@"
}

case ${LANE:?} in
panda)
  render mg/PandaOmron/TurnOnSinkFaucet/4demos_seed1/demo_gentex_im320_randcams.hdf5 -- \
    $CE/PandaOmron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_PandaOmron/2026-07-22-20-09-38/demo.hdf5
  render human/PandaOmron/TurnOnSinkFaucet/1demos_seed1/demo_gentex_im320_randcams.hdf5 -- \
    $CE/PandaOmron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_PandaOmron/2026-07-22-20-07-57/demo.hdf5
  render mg/PandaOmron/PnPCounterToSink/2demos_seed1/demo_gentex_im320_randcams.hdf5 obj -- \
    $CE/PandaOmron/pretrain/PickPlaceCounterToSink/PickPlaceCounterToSink_PandaOmron/2026-07-22-20-20-27/demo.hdf5
  render mg/PandaOmron/PnPCounterToSink/2demos_seed2/demo_gentex_im320_randcams.hdf5 obj -- \
    $CE/PandaOmron/pretrain/PickPlaceCounterToSink/PickPlaceCounterToSink_PandaOmron/2026-07-22-20-34-26/demo.hdf5
  ;;
ur5e)
  render mg/UR5eOmron/TurnOnSinkFaucet/2demos_seed1/demo_gentex_im320_randcams.hdf5 -- \
    $CE/UR5eOmron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_UR5eOmron/2026-07-22-20-09-41/demo.hdf5
  render mg/UR5eOmron/TurnOnSinkFaucet/3demos_seed2/demo_gentex_im320_randcams.hdf5 -- \
    $CE/UR5eOmron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_UR5eOmron/2026-07-22-20-21-36/demo.hdf5
  render mg/UR5eOmron/PnPCounterToSink/1demos_seed1/demo_gentex_im320_randcams.hdf5 obj -- \
    $CE/UR5eOmron/pretrain/PickPlaceCounterToSink/PickPlaceCounterToSink_UR5eOmron/2026-07-22-20-20-35/demo.hdf5
  render mg/UR5eOmron/PnPCounterToSink/1demos_seed2/demo_gentex_im320_randcams.hdf5 obj -- \
    $CE/UR5eOmron/pretrain/PickPlaceCounterToSink/PickPlaceCounterToSink_UR5eOmron/2026-07-22-20-34-23/demo.hdf5
  ;;
iiwa)
  render mg/IIWAOmron/TurnOnSinkFaucet/2demos_seed1/demo_gentex_im320_randcams.hdf5 -- \
    $CE/IIWAOmron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_IIWAOmron/2026-07-22-20-09-42/demo.hdf5
  render mg/IIWAOmron/TurnOnSinkFaucet/5demos_seed2/demo_gentex_im320_randcams.hdf5 -- \
    $CE/IIWAOmron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_IIWAOmron/2026-07-22-20-21-37/demo.hdf5
  ;;
kinova3)
  render mg/Kinova3Omron/TurnOnSinkFaucet/1demos_seed1/demo_gentex_im320_randcams.hdf5 -- \
    $CE/Kinova3Omron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_Kinova3Omron/2026-07-22-20-12-54/demo.hdf5
  render mg/Kinova3Omron/TurnOnSinkFaucet/3demos_seed2/demo_gentex_im320_randcams.hdf5 -- \
    $CE/Kinova3Omron/pretrain/TurnOnSinkFaucet/TurnOnSinkFaucet_Kinova3Omron/2026-07-22-20-21-38/demo.hdf5
  ;;
esac
echo "LANE=$LANE done"
