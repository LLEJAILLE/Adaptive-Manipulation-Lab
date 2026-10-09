"""Shaping par variations de potentiels, independant de MuJoCo.

Le calculateur ne detecte pas les contacts et ne cumule pas les scores : ces
responsabilites appartiennent a l'environnement. reset() memorise la distance
initiale ; calculate() avance une seule fois par decision. Les variations sont
signees : un aller-retour annule le shaping, sans annuler le cout du temps.
"""

import math

import numpy as np

NUM_EPISODES = 5
MAX_DURATION = 20.0
GRIPPER_OPEN = 0.045

def has_gripper_contact(contacts,cube_geom,gripper_geoms):
    """Reach succeeds on an active contact with a finger or gripper support."""
    return any(contact.dist <= 0 and contact.efc_address >= 0 and (
        contact.geom1 == cube_geom and contact.geom2 in gripper_geoms or
        contact.geom2 == cube_geom and contact.geom1 in gripper_geoms)
        for contact in contacts)


class RewardCalculator:
    def __init__(self, success_bonus=200.0, *, progress_scale=20.0, proximity_scale=20.0, proximity_radius=0.30, time_penalty=0.1, failure_penalty=-100.0):
        for name, value in (("success_bonus", success_bonus),
                            ("progress_scale", progress_scale),
                            ("proximity_scale", proximity_scale),
                            ("time_penalty", time_penalty)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} doit etre positif ou nul et fini.")
        if not math.isfinite(proximity_radius) or proximity_radius <= 0:
            raise ValueError("proximity_radius doit etre positif et fini.")
        if not math.isfinite(failure_penalty) or failure_penalty > 0:
            raise ValueError("failure_penalty doit etre negatif ou nul et fini.")
        self.success_bonus = float(success_bonus)
        self.progress_scale = float(progress_scale)
        self.proximity_scale = float(proximity_scale)
        self.proximity_radius = float(proximity_radius)
        self.time_penalty = float(time_penalty)
        self.failure_penalty = float(failure_penalty)
        self.previous_distance = None
        self.last_components = {}

    @staticmethod
    def _distance(gripper_pos, cube_pos):
        gripper_pos = np.asarray(gripper_pos, dtype=float)
        cube_pos = np.asarray(cube_pos, dtype=float)
        if gripper_pos.shape != (3,) or cube_pos.shape != (3,):
            raise ValueError("Les positions doivent etre des vecteurs 3D.")
        if not np.isfinite(gripper_pos).all() or not np.isfinite(cube_pos).all():
            raise ValueError("Les positions doivent etre finies.")
        return float(np.linalg.norm(gripper_pos - cube_pos))

    def proximity_potential(self, distance):
        """Potentiel lisse : amplitude maximale a zero, decroissance sur 30 cm."""
        ratio = distance / self.proximity_radius
        return self.proximity_scale * math.exp(-ratio * ratio)

    def reset(self, gripper_pos, cube_pos):
        """Initialiser la reference sans reward ni cout de decision."""
        self.previous_distance = self._distance(gripper_pos, cube_pos)
        self.last_components = {}

    def calculate(self, gripper_pos, cube_pos, success, *, failed=False):
        """Retourner (reward, distance) et exposer last_components.

        Appeler reset() au debut du scenario. Les bonus/penalites terminaux
        s'ajoutent au shaping et au cout de la derniere decision.
        """
        if self.previous_distance is None:
            raise RuntimeError("Appeler reset() avant la premiere decision.")
        if success and failed:
            raise ValueError("Une decision ne peut pas etre reussie et echouee.")
        distance = self._distance(gripper_pos, cube_pos)
        progress = self.progress_scale * (self.previous_distance - distance)
        proximity = (self.proximity_potential(distance)
                     - self.proximity_potential(self.previous_distance))
        terminal = (self.success_bonus if success else
                    self.failure_penalty if failed else 0.0)
        self.last_components = {
            "progress_reward": progress,
            "proximity_reward": proximity,
            "time_penalty": -self.time_penalty,
            "terminal_reward": terminal,
            "distance": distance,
        }
        reward = progress + proximity - self.time_penalty + terminal
        self.previous_distance = distance
        return reward, distance
