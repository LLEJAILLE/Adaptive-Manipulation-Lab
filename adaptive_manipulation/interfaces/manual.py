"""Interactive manual control with sliders and a gamepad."""

def run_manual(num_episodes=5, seed=42):

    import math
    import threading
    import time
    import tkinter as tk

    import mujoco
    import mujoco.viewer

    from adaptive_manipulation.interfaces.gamepad import JOINT_SPEED_DEG, PS5Controller
    from adaptive_manipulation.simulation.environment import Environment, GRIPPER_OPEN

    # Reglages de la session manuelle : meme seed = memes positions de depart.
    environment = Environment(seed=seed, num_episodes=num_episodes)
    DECISION_PHYSICS_STEPS = 10  # 50 decisions/s avec timestep=0.002 ; physique a 500 Hz.
    model = environment.model
    data = environment.data


    targets = [0.0, 0.0, 0.0]
    positions = [0.0, 0.0, 0.0]
    end_effector = [0.0, 0.0, 0.0]

    # Les glissieres de la pince sont commandees en metres, pas en radians.
    GRIPPER_CLOSED = 0.0
    gripper_target = GRIPPER_OPEN
    scenario_status = environment.status_text()
    scenario_episode = environment.episode
    session_done = False

    lock = threading.Lock()
    stop_event = threading.Event()

    joint_names = ["joint1", "joint2", "joint3"]
    joint_ids = [
        model.joint(name).id for name in joint_names
    ]
    site_id = model.site("end_effector").id
    arm_motor_ids = [model.actuator(name).id for name in ("motor1", "motor2", "motor3")]


    def simulation_loop():
        nonlocal gripper_target, scenario_status, scenario_episode, session_done
        with mujoco.viewer.launch_passive(model, data) as viewer:

            while viewer.is_running() and not stop_event.is_set():
                start = time.perf_counter()

                with lock:
                    commands = targets.copy()
                    opening = gripper_target

                # Environment gere la physique, le contact, le timer et la transition.
                result = None
                if not environment.is_done:
                    _, _, _, _, info = environment.step(
                        commands, opening, physics_steps=DECISION_PHYSICS_STEPS)
                    result = info.get("scenario_result")
                if result is not None:
                    outcome = "REUSSI" if result.success else "ECHEC"
                    print(f"Scenario {result.episode} : {outcome} ({result.reason}), "
                          f"{result.elapsed:.2f} s, cube={result.spawn_position}, "
                          f"reward={result.total_reward:.3f}, decisions={result.steps}, "
                          f"distance finale={result.final_distance:.3f} m")

                # Lecture des angles réels
                current_positions = [
                    float(data.qpos[model.jnt_qposadr[jid]])
                    for jid in joint_ids
                ]

                # Position de l'extrémité du robot (x, y, z)
                current_ee = data.site_xpos[site_id].copy()

                with lock:
                    if result is not None:
                        targets[:] = [0.0, 0.0, 0.0]
                        gripper_target = GRIPPER_OPEN
                    positions[:] = current_positions
                    end_effector[:] = current_ee.tolist()
                    scenario_status = environment.status_text()
                    scenario_episode = environment.episode
                    session_done = environment.is_done

                viewer.sync()

                elapsed = time.perf_counter() - start
                remaining = model.opt.timestep * DECISION_PHYSICS_STEPS - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        stop_event.set()


    root = tk.Tk()
    root.title("Adaptive Manipulation Lab - Control Panel")
    root.geometry("520x710")

    tk.Label(
        root,
        text="Robot Arm Controller",
        font=("Arial", 16, "bold")
    ).pack(pady=12)

    scenario_label = tk.Label(root, text=scenario_status, wraplength=490)
    scenario_label.pack(pady=6)

    joint_configs = [
        ("J1 - Base", -180, 180),
        ("J2 - Shoulder", -90, 90),
        ("J3 - Elbow", -135, 135),
    ]

    scales = []
    actual_labels = []


    def update_target(index, value):
        angle_rad = math.radians(float(value))
        with lock:
            targets[index] = angle_rad


    for i, (name, minimum, maximum) in enumerate(joint_configs):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=20, pady=6)

        tk.Label(frame, text=name).pack(anchor="w")

        scale = tk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            resolution=-1,
            digits=6,
            command=lambda value, idx=i: update_target(idx, value)
        )
        scale.pack(fill="x")
        scales.append(scale)

        label = tk.Label(frame, text="Actual: 0.0 deg")
        label.pack(anchor="e")
        actual_labels.append(label)


    position_label = tk.Label(root, text="End effector: (0, 0, 0)")
    position_label.pack(pady=8)


    def reset_robot():
        for scale in scales:
            scale.set(0)
        set_gripper(GRIPPER_OPEN)


    def set_gripper(opening):
        nonlocal gripper_target
        # Le bouton change la cible ; seul le thread de simulation ecrit data.ctrl.
        with lock:
            gripper_target = opening


    gripper_frame = tk.LabelFrame(root, text="Pince")
    gripper_frame.pack(fill="x", padx=20, pady=6)
    tk.Button(
        gripper_frame, text="Serrer la pince",
        command=lambda: set_gripper(GRIPPER_CLOSED)
    ).pack(side=tk.LEFT, expand=True, fill="x", padx=5, pady=8)
    tk.Button(
        gripper_frame, text="Relâcher la pince",
        command=lambda: set_gripper(GRIPPER_OPEN)
    ).pack(side=tk.LEFT, expand=True, fill="x", padx=5, pady=8)


    tk.Button(
        root,
        text="Reset targets",
        command=reset_robot
    ).pack(pady=6)


    gamepad = PS5Controller()
    gamepad_status = tk.Label(root, text=gamepad.status, wraplength=480)
    gamepad_status.pack(pady=4)
    tk.Label(
        root,
        text="Stick gauche : J1 / J2 | Stick droit vertical : J3\n"
             "Croix : serrer | Rond : relâcher | Triangle : reset du bras",
    ).pack(pady=4)
    last_gamepad_poll = time.perf_counter()
    displayed_episode = scenario_episode


    def poll_gamepad():
        nonlocal last_gamepad_poll, displayed_episode
        if stop_event.is_set():
            return
        now = time.perf_counter()
        # Eviter un saut de consigne si l'interface a ete temporairement bloquee.
        dt = min(now - last_gamepad_poll, 0.05)
        last_gamepad_poll = now
        controls = gamepad.poll()
        with lock:
            requested = targets.copy()
            episode = scenario_episode
            done = session_done
        if episode != displayed_episode:
            # Synchroniser les curseurs sans declencher leurs commandes Tkinter.
            for i, scale in enumerate(scales):
                callback = scale.cget("command")
                scale.configure(command="")
                scale.set(math.degrees(requested[i]))
                scale.configure(command=callback)
            displayed_episode = episode
        if done:
            gamepad_status.config(text=gamepad.status)
            root.after(20, poll_gamepad)
            return
        for i, axis in enumerate(controls.joint_axes):
            if axis == 0:
                continue
            jid = joint_ids[i]
            motor_id = arm_motor_ids[i]
            minimum = max(model.jnt_range[jid, 0], model.actuator_ctrlrange[motor_id, 0])
            maximum = min(model.jnt_range[jid, 1], model.actuator_ctrlrange[motor_id, 1])
            angle = requested[i] + math.radians(JOINT_SPEED_DEG) * axis * dt
            degrees = math.degrees(max(minimum, min(maximum, angle)))
            scales[i].set(degrees)
            # Mettre a jour immediatement la cible, sans attendre le callback Tkinter.
            update_target(i, degrees)
        if controls.close_gripper:
            set_gripper(GRIPPER_CLOSED)
        if controls.open_gripper:
            set_gripper(GRIPPER_OPEN)

        if controls.reset_targets:
            reset_robot()

        gamepad_status.config(text=gamepad.status)
        root.after(20, poll_gamepad)


    def refresh_ui():
        if stop_event.is_set():
            root.destroy()
            return

        with lock:
            actual = positions.copy()
            ee = end_effector.copy()
            status = scenario_status

        scenario_label.config(text=status)

        for i, angle in enumerate(actual):
            actual_labels[i].config(
                text=f"Actual: {math.degrees(angle):.1f} deg"
            )

        position_label.config(
            text=f"End effector: "
                 f"({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m"
        )

        root.after(100, refresh_ui)


    def on_close():
        stop_event.set()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_close)

    # La simulation tourne dans un thread séparé
    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()

    refresh_ui()
    poll_gamepad()
    try:
        root.mainloop()
    finally:
        stop_event.set()
        gamepad.close()
        thread.join(timeout=3)



def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation manual', description="Pilotage manuel du robot avec curseurs et manette")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes doit etre positif et seed non negative")
    run_manual(args.episodes, args.seed)
