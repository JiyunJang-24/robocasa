"""Convert mujoco_menagerie arm MJCFs into robosuite-compatible robot.xml.

Menagerie models are standalone MuJoCo scenes; robosuite needs a bare arm
ending in a `right_hand` mount body, driven by torque `<motor>` actuators,
with no reliance on MJCF features robosuite's own XML layer doesn't implement.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

MEN = Path("/data1/local/robots_render/mujoco_menagerie")
OUT = Path("/data1/local/robocasa/custom_robots")

# Element tags that participate in MJCF default-class resolution.
_DEFAULTABLE = {
    "geom",
    "joint",
    "site",
    "camera",
    "light",
    "mesh",
    "material",
    "position",
    "motor",
    "general",
    "velocity",
    "equality",
    "pair",
    "inertial",
    "body",
    "freejoint",
    "tendon",
}

SPECS = {
    "piper": dict(
        src=MEN / "agilex_piper/piper.xml",
        root_body="base_link",
        finger_bodies=["link7", "link8"],
        hand_parent="link6",
        # Tool flange, i.e. where link6's own geometry ends (z-extent tops out
        # at 0.067). NOT the 0.13503 that link7/link8 mount at: those finger
        # meshes reach back over the palm, so stripping them and mounting at
        # 0.13503 left the gripper floating 68mm off the end of the wrist.
        hand_pos="0 0 0.067",
        hand_quat="1 0 0 0",
        torque={
            "joint1": 100,
            "joint2": 100,
            "joint3": 100,
            "joint4": 30,
            "joint5": 30,
            "joint6": 30,
        },
    ),
    "yam": dict(
        src=MEN / "i2rt_yam/yam.xml",
        root_body="arm",
        finger_bodies=["link_left_finger", "link_right_finger"],
        hand_parent="link_6",
        # Tool flange (link_6 geometry ends at z=0.087). The original 0.1347
        # is yam.xml's grasp_site -- the point between the closed fingertips,
        # 48mm beyond the wrist -- which left a visible gap once the finger
        # bodies were stripped.
        hand_pos="0 0 0.087",
        hand_quat="1 0 0 -1",
        torque={
            "joint1": 28,
            "joint2": 28,
            "joint3": 28,
            "joint4": 10,
            "joint5": 10,
            "joint6": 10,
        },
    ),
}


def collect_defaults(default_el, inherited=None, out=None, name="__main__"):
    """Flatten nested <default> into {class_name: {tag: {attr: val}}}.

    MJCF nests default classes and each inherits its parent's attributes;
    robosuite's _get_default_classes only walks the top level and ignores
    inheritance entirely, so resolve the whole thing here instead.
    """
    out = {} if out is None else out
    merged = {tag: dict(attrs) for tag, attrs in (inherited or {}).items()}
    for child in default_el:
        if child.tag == "default":
            continue
        merged.setdefault(child.tag, {}).update(child.attrib)
    out[name] = merged
    for child in default_el:
        if child.tag == "default":
            collect_defaults(child, merged, out, child.get("class"))
    return out


def bake_defaults(el, defaults, childclass="__main__"):
    """Write resolved default attributes onto every element, in place."""
    cls = el.attrib.pop("class", None) or childclass
    if el.tag in _DEFAULTABLE and cls in defaults:
        for attr, val in defaults[cls].get(el.tag, {}).items():
            el.attrib.setdefault(attr, val)
    childclass = el.attrib.pop("childclass", None) or cls
    for child in el:
        bake_defaults(child, defaults, childclass)


def find_parent(root, child):
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None


def convert(name, spec):
    tree = ET.parse(spec["src"])
    root = tree.getroot()
    root.set("model", f"{name}_robosuite")

    # robosuite's resolve_asset_dependency() rewrites every asset `file` to
    # an absolute path using the *xml's own folder* and ignores <compiler
    # meshdir> entirely, so mirror its own robots' layout: assets in a
    # subfolder next to robot.xml, referenced by relative path.
    dst_dir = OUT / name
    asset_dst = dst_dir / "assets"
    if asset_dst.exists():
        shutil.rmtree(asset_dst)
    asset_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(spec["src"].parent / "assets", asset_dst)
    comp = root.find("compiler")
    if comp is None:  # NB: an Element with no children is falsy
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.attrib.pop("meshdir", None)
    comp.set("angle", "radian")
    for node in root.iter():
        f = node.get("file")
        if f is not None:
            node.set("file", f"assets/{Path(f).name}")

    # MuJoCo lets a <mesh> stay anonymous and auto-names it after its file,
    # but robosuite namespaces a robot by rewriting explicit name= attributes
    # (and every geom's mesh= reference) to `robot0_*`. An anonymous mesh
    # keeps its bare auto-name while the geoms that use it get prefixed, so
    # the compile fails with "mesh 'robot0_link2_0' not found". Name them all
    # up front so both sides get renamed together.
    seen = set()
    for mesh in root.iter("mesh"):
        if mesh.get("name"):
            seen.add(mesh.get("name"))
            continue
        stem = Path(mesh.get("file")).stem
        mesh_name, dedup = stem, 1
        while mesh_name in seen:  # distinct files can share a stem
            mesh_name, dedup = f"{stem}_{dedup}", dedup + 1
        mesh.set("name", mesh_name)
        seen.add(mesh_name)

    # Resolve + inline every default, then drop the section so robosuite's
    # partial re-implementation of the same mechanism has nothing to trip on.
    default_el = root.find("default")
    if default_el is not None:
        defaults = collect_defaults(default_el)
        bake_defaults(root, defaults)
        root.remove(default_el)

    # robosuite only understands geom group 0 (collision) / 1 (visual); the
    # menagerie convention here is 2=visual, 3=collision, which makes its
    # _add_default_name_filter raise KeyError on the raw value. Re-derive the
    # group from what actually decides collision behaviour rather than
    # hardcoding 2->1/3->0, so any other group value converts correctly too.
    for geom in root.iter("geom"):
        visual = geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0"
        geom.set("group", "1" if visual else "0")

    for tag in ("keyframe", "equality"):
        for el in root.findall(tag):
            root.remove(el)
    wb = root.find("worldbody")
    for light in wb.findall("light"):
        wb.remove(light)

    for fb in spec["finger_bodies"]:
        for body in list(root.iter("body")):
            if body.get("name") == fb:
                find_parent(root, body).remove(body)

    act = root.find("actuator")
    for el in list(act):
        act.remove(el)
    for joint, t in spec["torque"].items():
        ET.SubElement(
            act,
            "motor",
            {
                "ctrllimited": "true",
                "ctrlrange": f"-{t}.0 {t}.0",
                "joint": joint,
                "name": f"torq_{joint}",
            },
        )

    for body in root.iter("body"):
        if body.get("name") == spec["hand_parent"]:
            hand = ET.SubElement(
                body,
                "body",
                {
                    "name": "right_hand",
                    "pos": spec["hand_pos"],
                    "quat": spec["hand_quat"],
                },
            )
            ET.SubElement(
                hand,
                "site",
                {
                    "name": "right_center",
                    "pos": "0 0 0",
                    "size": "0.01",
                    "rgba": "1 0.3 0.3 0",
                    "group": "1",
                },
            )
            break

    # robosuite expects the robot's root body to be literally "base"; fix up
    # every other reference to the old name too (piper's <contact><exclude>
    # pair, which otherwise fails to resolve after the rename).
    old_root = spec["root_body"]
    for body in wb.findall("body"):
        if body.get("name") == old_root:
            body.set("name", "base")
    for el in root.iter():
        for attr in ("body", "body1", "body2"):
            if el.get(attr) == old_root:
                el.set(attr, "base")

    dst = dst_dir / "robot.xml"
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="unicode")

    leftover_cls = [
        e.tag for e in root.iter() if "class" in e.attrib or "childclass" in e.attrib
    ]
    joints = [j.get("name") for b in root.iter("body") for j in b.findall("joint")]
    print(f"[{name}] wrote {dst}")
    print(f"[{name}] joints={joints} leftover_class_attrs={leftover_cls}")


for n, s in SPECS.items():
    convert(n, s)
