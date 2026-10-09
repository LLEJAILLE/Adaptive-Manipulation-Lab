"""Scenario engine shared by manual control and the Gymnasium AI adapter.

La scene et les scenarios sont geres ici ; l'affichage et la manette restent dans
interfaces/manual.py. Une meme seed reproduit la sequence des positions initiales,
quels que soient les actions et les resultats. gym_env.py adapte ce moteur a Gymnasium.

    env = Environment(seed=42, num_episodes=5, max_steps=1000)
    observation, reward, terminated, truncated, info = env.step(
        [0.0, 0.0, 0.0], gripper_opening=0.045, physics_steps=10)

step() passe automatiquement au scenario suivant apres un contact ou un timeout.
Apres le dernier, is_done vaut True et la scene reste figee. Le timer utilise le
temps reel monotone en manuel ; en IA/machine, il utilise le temps physique MuJoCo.
clock peut etre remplace pour les tests manuels sans attente.
Chaque appel a step est une decision : une seule reward est calculee, quel que
soit le nombre de pas physiques. Au retour d'une fin, observation et info restent
ceux du scenario termine ; info['next_observation'] contient le depart du suivant.
"""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco

from adaptive_manipulation.tasks.reach import RewardCalculator, NUM_EPISODES, MAX_DURATION, GRIPPER_OPEN, has_gripper_contact
from adaptive_manipulation.core.paths import MODEL_PATH

@dataclass(frozen=True)
class ScenarioResult:
    """Resultat conserve avant la remise a zero de la scene."""

    episode: int
    success: bool
    reason: str  # "contact", "timeout" ou "step_limit"
    elapsed: float
    spawn_position: tuple
    reward: float
    total_reward: float
    steps: int
    final_distance: float
    seed: int | None


class Environment:
    """Une serie de scenarios. Appeler step() depuis un seul thread.

    seed : generateur aleatoire local, sans effet sur random global.
    render_mode : "human" pour le viewer externe, "machine" ou None sans GUI.
    mode : "manual" (timer reel) ou "ai" (timer simule).
    num_episodes : nombre de scenarios avant la fin de la session.
    xml_path : scene a charger ; par defaut models/robot_arm.xml.
    max_steps : limite de decisions, ou None pour garder seulement le timer.
    Coefficients de shaping : success_bonus, failure_penalty, progress_scale,
    proximity_scale, proximity_radius (metres), time_penalty (cout positif).
    """

    def __init__(self, seed=42, render_mode="human", mode="manual", *, num_episodes=NUM_EPISODES, xml_path=None, clock=time.monotonic, max_steps=None, success_bonus=200.0, failure_penalty=-100.0, progress_scale=20.0, proximity_scale=20.0, proximity_radius=0.30, time_penalty=0.1):
        if mode not in ("manual", "ai"):
            raise ValueError("mode doit etre 'manual' ou 'ai'.")
        if render_mode not in ("human", "machine", None):
            raise ValueError("render_mode doit etre 'human', 'machine' ou None.")
        if isinstance(num_episodes, bool) or not isinstance(num_episodes, int) or num_episodes < 1:
            raise ValueError("num_episodes doit etre un entier positif.")
        if not math.isfinite(MAX_DURATION) or MAX_DURATION <= 0:
            raise ValueError("MAX_DURATION doit etre une duree positive et finie.")
        if max_steps is not None and (
                isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps doit etre un entier positif ou None.")

        self.seed = seed
        self.rng = random.Random(seed)
        self.render_mode = render_mode
        self.mode = mode
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.reward_calculator = RewardCalculator(
            success_bonus, failure_penalty=failure_penalty,
            progress_scale=progress_scale, proximity_scale=proximity_scale,
            proximity_radius=proximity_radius, time_penalty=time_penalty)
        self._clock = clock
        path = xml_path or MODEL_PATH
        self.xml_path = str(Path(path).resolve())
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self._arm_motors = [self.model.actuator(f"motor{i}").id for i in (1, 2, 3)]
        self._gripper_motors = [self.model.actuator(f"gripper_motor_{side}").id
                                for side in ("left", "right")]
        self._gripper_qpos = [self.model.joint(f"gripper_{side}").qposadr[0]
                              for side in ("left", "right")]
        self._cube_geom = self.model.geom("cube_geom").id
        self._cube_qpos = self.model.joint("cube_joint").qposadr[0]
        self._zone_geom = self.model.geom("start_zone_geom").id
        self._gripper_site = self.model.site("end_effector").id

        # Inclure le support et les doigts de la pince, mais pas l'avant-bras.
        gripper_body = self.model.body("gripper").id
        self._gripper_geoms = set()
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            while body_id != 0:
                if body_id == gripper_body:
                    self._gripper_geoms.add(geom_id)
                    break
                body_id = self.model.body_parentid[body_id]

        self.episode = 0
        self.is_done = False
        self.results = []
        self.spawn_position = ()
        self._started_at = 0.0
        self._simulation_started_at = 0.0
        self._start_scenario()

    @property
    def elapsed(self):
        """Temps reel du scenario courant ; fixe a la fin de la session."""
        if self.is_done:
            return self.results[-1].elapsed
        if self.mode == "ai" or self.render_mode == "machine":
            return max(0.0, float(self.data.time) - self._simulation_started_at)
        return max(0.0, self._clock() - self._started_at)

    @property
    def remaining(self):
        return max(0.0, MAX_DURATION - self.elapsed)

    def _start_scenario(self):
        """Effacer positions, vitesses, forces et commandes puis placer le cube."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._gripper_qpos] = GRIPPER_OPEN
        self.data.ctrl[self._gripper_motors] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

        zone_size = self.model.geom_size[self._zone_geom]
        cube_size = self.model.geom_size[self._cube_geom]
        # Marge de 5 mm : toute la base du cube reste dans la zone.
        limits = zone_size[:2] - cube_size[:2] - 0.005
        if min(limits) < 0:
            raise ValueError("La zone start est trop petite pour le cube.")
        local_position = [
            self.rng.uniform(-limits[0], limits[0]),
            self.rng.uniform(-limits[1], limits[1]),
            zone_size[2] + cube_size[2] + 0.001,
        ]
        rotation = self.data.geom_xmat[self._zone_geom].reshape(3, 3)
        position = self.data.geom_xpos[self._zone_geom] + rotation @ local_position
        self.data.qpos[self._cube_qpos:self._cube_qpos + 3] = position
        # Aligner le cube avec la zone, y compris si elle est orientee dans le XML.
        mujoco.mju_mat2Quat(
            self.data.qpos[self._cube_qpos + 3:self._cube_qpos + 7], rotation.ravel()
        )
        mujoco.mj_forward(self.model, self.data)
        self.spawn_position = tuple(float(value) for value in position)
        self.episode += 1
        self._started_at = self._clock()
        self._simulation_started_at = float(self.data.time)
        self.reward = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.success = False
        self.final_distance = None
        self.reward_calculator.reset(
            self.data.site_xpos[self._gripper_site], self.data.geom_xpos[self._cube_geom])

    def _cube_touched(self):
        """Contact physique MuJoCo, sans seuil de distance approximatif au site."""
        return has_gripper_contact(self.data.contact,self._cube_geom,self._gripper_geoms)

    def _observation(self):
        """Copies des positions et de l'etat, sans alias sur les donnees MuJoCo."""
        return {
            "gripper_pos": self.data.site_xpos[self._gripper_site].copy(),
            "cube_pos": self.data.geom_xpos[self._cube_geom].copy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def step(self, arm_targets, gripper_opening, *, physics_steps=1):
        """Une decision -> observation, reward, terminated, truncated, info.

        Apres une fin, le scenario suivant est deja initialise au retour de step().
        L'appelant doit aussi remettre ses cibles manuelles a zero et ouvrir la
        pince, pour ne pas reappliquer les commandes du scenario precedent.
        Un contact apres la deadline ne compte pas comme une reussite.
        terminated = contact ou echec a l'echeance/limite de decisions.
        Ces limites font partie de la tache ; elles ne sont pas des troncatures.
        La reward est calculee une seule fois, apres au plus physics_steps pas.
        L'observation et les metriques de fin sont captures avant l'autoreset.
        """
        if self.is_done:
            raise RuntimeError("Session terminee : creer un nouvel Environment pour rejouer.")
        if (isinstance(physics_steps, bool) or not isinstance(physics_steps, int)
                or physics_steps < 1):
            raise ValueError("physics_steps doit etre un entier positif.")
        self.data.ctrl[self._arm_motors] = arm_targets
        self.data.ctrl[self._gripper_motors] = gripper_opening
        success = False
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
            elapsed = self.elapsed
            success = self._cube_touched() and elapsed <= MAX_DURATION
            if success or elapsed >= MAX_DURATION:
                break
        # mj_step integre qpos apres le calcul des sites ; actualiser les positions
        # pour mesurer la distance correspondant a l'etat retourne.
        mujoco.mj_forward(self.model, self.data)
        observation = self._observation()
        # Determiner la fin avant la reward, pour appliquer une seule fois le
        # bonus de contact ou la penalite de timeout/limite de decisions.
        elapsed = self.elapsed
        next_steps = self.steps + 1
        step_limit = self.max_steps is not None and next_steps >= self.max_steps
        failed = not success and (elapsed >= MAX_DURATION or step_limit)
        terminated = success or failed
        truncated = False
        self.reward, self.final_distance = self.reward_calculator.calculate(
            observation["gripper_pos"], observation["cube_pos"], success,
            failed=failed)
        self.total_reward += self.reward
        self.steps = next_steps
        self.success = success
        reason = ("contact" if success else "timeout" if elapsed >= MAX_DURATION
                  else "step_limit" if step_limit else None)
        info = {
            "episode": self.episode, "seed": self.seed, "success": success,
            "reward": self.reward, "total_reward": self.total_reward,
            "steps": self.steps, "final_distance": self.final_distance,
            "elapsed": elapsed, "reason": reason,
            **self.reward_calculator.last_components,
        }
        reward = self.reward
        if terminated or truncated:
            result = ScenarioResult(
                self.episode, success, reason, elapsed, self.spawn_position,
                reward, self.total_reward, self.steps, self.final_distance, self.seed)
            self.results.append(result)
            info["scenario_result"] = result
            if self.episode >= self.num_episodes:
                self.is_done = True
            else:
                self._start_scenario()
                info["next_observation"] = self._observation()
        return observation, reward, terminated, truncated, info

    def status_text(self):
        """Texte du panneau manuel, a lire dans le meme thread que step()."""
        successes = sum(result.success for result in self.results)
        if self.is_done:
            return (f"Session terminee : {successes}/{self.num_episodes} reussites, "
                    f"{self.num_episodes - successes} echecs")
        text = (f"Scenario {self.episode}/{self.num_episodes} | "
                f"Temps restant : {self.remaining:.1f} s | Reussites : {successes}")
        if self.results:
            last = self.results[-1]
            outcome = ("reussi (contact)" if last.success else
                       "echoue (limite de decisions)" if last.reason == "step_limit" else
                       "echoue (temps ecoule)")
            text += f"\nPrecedent : scenario {last.episode} {outcome}"
        return text
