# RoboCasa human-demo cross-embodiment regeneration

Replays RoboCasa v1.0 atomic **human** demos (LeRobot + `extras/`) on other robots
(UR5eOmron / IIWAOmron / Kinova3Omron) by OSC tracking of the source EEF trajectory,
and writes LeRobot v2.1 datasets (3 cams, 256x256, 20 fps).

Spec:
- restore sim from `extras/episode_*/{model.xml.gz, states.npz, ep_meta.json}`
- target gripper: robosuite stock `robotiq_gripper_85.xml` registered as `Robotiq85StockGripper`
  (finger kp 40, pad friction 2.0)
- OSC EEF pose tracking, gripper command copied from source
- base back-off <= 30 cm when unreachable, finger back-off <= 3 cm
- keep only episodes with `env.is_success()`
- jump filter: reject episode if any step moves base > 5 cm or EEF > 10 cm
- output extras + `regen_info.json` (source episode, base offset)

Rewritten on poipu 2026-11-14 after the lanai copy was lost (lanai down).
