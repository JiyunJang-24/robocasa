"""Download RoboCasa v1.0 atomic human LeRobot tars (with extras/) for given tasks."""
import argparse, json, os, subprocess, tarfile
import robocasa

p = argparse.ArgumentParser()
p.add_argument("--tasks", nargs="+", required=True)
p.add_argument("--out", default=os.path.join(os.path.dirname(robocasa.__path__[0]), "datasets", "v1.0"))
a = p.parse_args()
links = json.load(open(os.path.join(robocasa.__path__[0], "models/assets/box_links/box_links_ds.json")))
for t in a.tasks:
    keys = [k for k in links if k.startswith(f"pretrain/atomic/{t}/") and k.endswith("/lerobot.tar") and "/mg/" not in k]
    assert len(keys) == 1, (t, keys)
    k = keys[0]
    dst = os.path.join(a.out, os.path.dirname(k))
    if os.path.isdir(os.path.join(dst, "lerobot", "extras")):
        print("skip", t); continue
    os.makedirs(dst, exist_ok=True)
    u = links[k]; url = u.split("/s/")[0] + "/shared/static/" + u.rstrip("/").split("/")[-1] + ".tar"
    tar = os.path.join(dst, "lerobot.tar")
    subprocess.run(["curl", "-sSL", "--retry", "5", "-o", tar, url], check=True)
    with tarfile.open(tar) as tf:
        tf.extractall(dst, filter="data")
    os.remove(tar)
    print("ok", t, dst)
