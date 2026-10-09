"""Record 30 gamepad scenarios using the exact SAC observation/action contract."""

import argparse
import json
from pathlib import Path

from adaptive_manipulation.core.config import TrainingConfig
from adaptive_manipulation.interfaces.actions import normalized_gamepad_action
from adaptive_manipulation.core.paths import TRAINING_CONFIG, DEMONSTRATIONS_DIR




def run_recording(config, output, episodes=30, seed_start=3_000_000, resume=False):
    import threading
    import time
    import traceback
    import tkinter as tk
    from tkinter import messagebox

    import mujoco.viewer
    import numpy as np

    from adaptive_manipulation.data.demonstrations import DemoRecorder
    from adaptive_manipulation.simulation.environment import GRIPPER_OPEN
    from adaptive_manipulation.interfaces.gamepad import PS5Controller
    from adaptive_manipulation.simulation.gym_env import ManipulationEnv

    env = ManipulationEnv(render_mode='machine', **config.environment_kwargs())
    try:
        recorder = DemoRecorder(output, env, episodes, seed_start, resume)
    except Exception:
        env.close()
        raise
    if recorder.done:
        print(f'Session deja terminee : {recorder.successes}/{episodes} reussites. {output}')
        env.close()
        return
    observation, reset_info = env.reset(seed=recorder.next_seed)
    lock = threading.Lock()
    stop_event, start_event = threading.Event(), threading.Event()
    state = dict(ready=True, running=False, finished=False, connected=False,
                 axes=(0.,0.,0.), gripper_goal=GRIPPER_OPEN, resetting=False,
                 status=f"Scenario {len(recorder.completed)+1}/{episodes} — seed {recorder.next_seed}\nCliquez sur Demarrer lorsque vous etes pret.",
                 error=None)
    gamepad = PS5Controller()
    root = tk.Tk()
    root.title('Adaptive Manipulation Lab — Demonstrations')
    root.geometry('610x420')
    tk.Label(root, text='Enregistrement des demonstrations', font=('Arial',16,'bold')).pack(pady=12)
    status_label = tk.Label(root, text=state['status'], wraplength=575, font=('Arial',12))
    status_label.pack(pady=15)
    controller_label = tk.Label(root, text=gamepad.status, wraplength=575)
    controller_label.pack(pady=8)
    tk.Label(root, text='Stick gauche : J1 / J2 | Stick droit vertical : J3\n'
             'Croix : fermer | Rond : ouvrir | Triangle : retour progressif au depart\n'
             '20 secondes simulees maximum par scenario. Le contact suffit pour reussir.',
             wraplength=575).pack(pady=8)

    def start():
        with lock:
            if state['ready'] and state['connected']:
                state['ready'] = False
                start_event.set()

    start_button = tk.Button(root, text='Demarrer le scenario', command=start)
    start_button.pack(pady=10)
    tk.Label(root, text=f'Sauvegarde automatique : {Path(output).resolve()}', wraplength=575).pack(pady=8)

    def simulation_loop():
        nonlocal observation, reset_info
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running() and not stop_event.is_set():
                    start_time = time.perf_counter()
                    if start_event.is_set():
                        start_event.clear()
                        recorder.begin(recorder.next_seed, reset_info['spawn_position'])
                        with lock:
                            state.update(running=True, gripper_goal=GRIPPER_OPEN, resetting=False)
                    with lock:
                        running, connected = state['running'], state['connected']
                        axes = state['axes']
                        goal, resetting = state['gripper_goal'], state['resetting']
                    if running and connected:
                        action = normalized_gamepad_action(axes,goal,env.targets,env.increments,resetting)
                        if resetting:
                            desired = np.array([0.,0.,0.,GRIPPER_OPEN])
                            if np.allclose(env.targets, desired, atol=1e-7, rtol=0):
                                with lock:
                                    state['resetting'] = False
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        recorder.add(observation,action,reward,next_obs,terminated,truncated,info)
                        observation = next_obs
                        text = (f'Scenario {len(recorder.completed)+1}/{episodes} — seed {recorder.next_seed}\n'
                                f'Temps restant : {env.core.remaining:.1f} s | Distance : {info["distance"]:.3f} m\n'
                                f'Reussites enregistrees : {recorder.successes}')
                        if terminated or truncated:
                            saved = recorder.finish(info)
                            outcome = 'REUSSI' if saved['success'] else 'ECHEC'
                            print(f"Scenario {saved['episode']}/{episodes} | seed {saved['seed']} | {outcome} "
                                  f"({saved['reason']}) | {saved['steps']} decisions", flush=True)
                            text = f"Precedent : {outcome} — {saved['reason']}\n"
                            if recorder.done:
                                text += f'Session terminee : {recorder.successes}/{episodes} reussites.\nVous pouvez fermer les fenetres.'
                            else:
                                observation, reset_info = env.reset(seed=recorder.next_seed)
                                text += f'Scenario {len(recorder.completed)+1}/{episodes} — seed {recorder.next_seed}\nCliquez sur Demarrer lorsque vous etes pret.'
                            with lock:
                                state.update(running=False, ready=not recorder.done, finished=recorder.done,
                                             axes=(0.,0.,0.), gripper_goal=GRIPPER_OPEN, resetting=False)
                        with lock:
                            state['status'] = text
                    viewer.sync()
                    stop_event.wait(max(0.,env.decision_dt-(time.perf_counter()-start_time)))
        except Exception as error:
            traceback.print_exc()
            with lock:
                state['error'] = str(error)
        finally:
            try:
                recorder.interrupt()
            except Exception as error:
                traceback.print_exc()
                with lock:
                    state['error'] = f'Erreur de sauvegarde : {error}'
            env.close()
            stop_event.set()

    def poll():
        if stop_event.is_set():
            return
        controls = gamepad.poll()
        with lock:
            state['connected'] = gamepad.device is not None
            state['axes'] = controls.joint_axes
            if controls.close_gripper:
                state['gripper_goal'] = 0.
            if controls.open_gripper:
                state['gripper_goal'] = GRIPPER_OPEN
            if controls.reset_targets:
                state.update(resetting=True, gripper_goal=GRIPPER_OPEN)
            elif any(controls.joint_axes):
                state['resetting'] = False
        controller_label.config(text=gamepad.status)
        root.after(20,poll)

    def refresh():
        with lock:
            snapshot = state.copy()
        if stop_event.is_set():
            if snapshot['error']:
                messagebox.showerror('Enregistrement interrompu', snapshot['error'], parent=root)
            root.destroy()
            return
        text = snapshot['status']
        if snapshot['running'] and not snapshot['connected']:
            text += '\nManette deconnectee : simulation en pause jusqu a la reconnexion.'
        status_label.config(text=text)
        start_button.config(state=tk.NORMAL if snapshot['ready'] and snapshot['connected'] else tk.DISABLED)
        root.after(100,refresh)

    def close():
        stop_event.set()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW',close)
    thread = threading.Thread(target=simulation_loop,daemon=False)
    thread.start()
    poll()
    refresh()
    try:
        root.mainloop()
    finally:
        stop_event.set()
        gamepad.close()
        # Saving is finished before process exit; no daemon thread can drop data.
        thread.join()
    print(f'Scenarios sauvegardes : {len(recorder.completed)}/{episodes}, reussites : {recorder.successes}',flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation record', description=__doc__)
    parser.add_argument('--config', default=str(TRAINING_CONFIG))
    parser.add_argument('--output', default=str(DEMONSTRATIONS_DIR/'gamepad_30'))
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed-start', type=int, default=3_000_000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.episodes > 9999 or args.seed_start < 0:
        parser.error('Use 1..9999 episodes and a nonnegative seed-start')
    config = TrainingConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8')))
    run_recording(config,args.output,args.episodes,args.seed_start,args.resume)


if __name__ == '__main__':
    main()
