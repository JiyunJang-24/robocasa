"""Recover published-episode -> atomic_new-source-episode mapping EXACTLY.

visual_robust_new's writer copies the source episode's raw_actions verbatim
into the published `action` column (see CrossEmbodimentLerobotWriter.add_episode:
`"action": list(raw_actions.astype(np.float32))`). So the action array is a
bit-exact fingerprint of which source episode a published episode came from --
no tolerance, no kinematics, no ambiguity. Measured: correct candidate gives
max|diff| == 0.0, wrong candidates 2.0.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(os.environ.get("ROBOCASA_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "datagen"))
import generate_visual_robust_cross_embodiment_x as gen  # noqa: E402

ATOMIC_NEW_ROOT = Path(
    os.environ.get(
        "ATOMIC_NEW_ROOT",
        REPO_ROOT / "third_party" / "barx" / "datasets" / "atomic_new",
    )
)
COMBOS = {
    "IIWAOmron_PnPCounterToSink": ATOMIC_NEW_ROOT
    / "IIWAOmron/pretrain/PnPCounterToSink/lerobot",
    "PandaOmron_TurnOnSinkFaucet": ATOMIC_NEW_ROOT
    / "PandaOmron/pretrain/TurnOnSinkFaucet/lerobot",
    "UR5eOmron_PnPSinkToCounter": ATOMIC_NEW_ROOT
    / "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
}
PUB_ROOT = Path(
    os.environ.get(
        "VISUAL_ROBUST_NEW_DIR", REPO_ROOT / "datasets" / "visual_robust_new"
    )
)
OUT_DIR = Path(
    os.environ.get(
        "MAPPING_OUT_DIR",
        REPO_ROOT
        / "outputs"
        / "datagen"
        / "visual_robust_new_segmentation"
        / "mapping",
    )
)


def load_meta(path):
    eps = {}
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        eps[rec["episode_index"]] = (rec["tasks"][0], rec["length"])
    return eps


def source_raw_actions(src_dir: Path, idx: int) -> np.ndarray:
    chunk = idx // 1000
    df = pd.read_parquet(
        src_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"
    )
    lerobot_act = np.stack(df["action"].to_numpy()).astype(np.float64)
    return gen.lerobot_action_to_raw(lerobot_act).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_ok = True
    for combo, src_dir in COMBOS.items():
        src_dir = Path(src_dir)
        src_eps = load_meta(src_dir / "meta" / "episodes.jsonl")
        pub_eps = load_meta(PUB_ROOT / combo / "lerobot" / "meta" / "episodes.jsonl")

        key_to_src = defaultdict(list)
        for idx, key in src_eps.items():
            key_to_src[key].append(idx)

        mapping, unresolved = {}, []
        used = {}
        for pub_idx in sorted(pub_eps):
            key = pub_eps[pub_idx]
            cands = key_to_src.get(key, [])
            pub_df = pd.read_parquet(
                PUB_ROOT
                / combo
                / "lerobot"
                / "data"
                / "chunk-000"
                / f"episode_{pub_idx:06d}.parquet"
            )
            pub_act = np.stack(pub_df["action"].to_numpy()).astype(np.float32)

            exact = []
            for c in cands:
                ra = source_raw_actions(src_dir, c)
                if ra.shape == pub_act.shape and np.array_equal(ra, pub_act):
                    exact.append(c)
            if len(exact) == 1:
                mapping[pub_idx] = exact[0]
                if exact[0] in used:
                    unresolved.append(
                        (
                            pub_idx,
                            f"source {exact[0]} already claimed by pub {used[exact[0]]}",
                        )
                    )
                used[exact[0]] = pub_idx
            elif len(exact) == 0:
                unresolved.append(
                    (pub_idx, f"no exact action match among {len(cands)} candidates")
                )
            else:
                # identical action arrays -> physically identical episodes; any
                # is equally valid as a render source
                mapping[pub_idx] = exact[0]
                print(
                    f"  note pub {pub_idx}: {len(exact)} sources have identical actions {exact[:4]}, using {exact[0]}"
                )

        status = "OK" if not unresolved else "INCOMPLETE"
        print(f"=== {combo}: resolved {len(mapping)}/{len(pub_eps)} [{status}] ===")
        for u in unresolved[:10]:
            print("   unresolved:", u)
        if unresolved:
            overall_ok = False
        identity = sum(1 for k, v in mapping.items() if k == v)
        print(
            f"   identity (pub==src): {identity}, remapped: {len(mapping) - identity}"
        )

        (OUT_DIR / f"{combo}.json").write_text(
            json.dumps({"mapping": mapping, "unresolved": unresolved}, indent=1)
        )
        print(f"   wrote {OUT_DIR / f'{combo}.json'}")

    print("ALL_RESOLVED" if overall_ok else "SOME_UNRESOLVED")


if __name__ == "__main__":
    main()
