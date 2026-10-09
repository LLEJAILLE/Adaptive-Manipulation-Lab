"""Lecture d'une DualSense via SDL, independante de MuJoCo et de Tkinter."""

import os
import time
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Continuer a lire les sticks lorsque la fenetre MuJoCo a le focus.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

try:
    import pygame
    from pygame._sdl2 import controller
except ImportError:
    pygame = None
    controller = None


DEADZONE = 0.15
JOINT_SPEED_DEG = 50.0  # Vitesse maximale des articulations.


def normalize_axis(raw, deadzone=DEADZONE):
    """Supprimer la derive au repos et conserver une progression douce."""
    value = max(-1.0, min(1.0, raw / 32767.0))
    if abs(value) <= deadzone:
        return 0.0
    magnitude = (abs(value) - deadzone) / (1.0 - deadzone)
    return magnitude if value > 0 else -magnitude


@dataclass(frozen=True)
class GamepadInput:
    joint_axes: tuple = (0.0, 0.0, 0.0)
    close_gripper: bool = False
    open_gripper: bool = False
    reset_targets: bool = False


class PS5Controller:
    """Appeler poll depuis le thread de l'interface, environ toutes les 20 ms."""

    def __init__(self):
        self.device = None
        self._previous_buttons = (False, False, False)
        self._next_scan = 0.0
        self._ready = False
        self.status = "Manette : pygame absent (pip install -r requirements.txt)"
        if pygame is None:
            return
        try:
            # Initialiser les evenements sans ouvrir de fenetre ni activer l'audio.
            pygame.display.init()
            controller.init()
            self._ready = True
            self.status = "Manette : recherche..."
        except pygame.error as error:
            self.status = f"Manette indisponible : {error}"

    def _scan(self):
        indices = [i for i in range(controller.get_count()) if controller.is_controller(i)]
        # Donner la priorite a la DualSense si plusieurs manettes sont branchees.
        indices.sort(key=lambda i: not any(
            word in (controller.name_forindex(i) or "").lower()
            for word in ("dualsense", "ps5")
        ))
        if not indices:
            self.status = "Manette : aucune detectee (reconnexion automatique)"
            return
        self.device = controller.Controller(indices[0])
        # Un bouton deja maintenu au branchement ne declenche pas de commande.
        self._previous_buttons = self._buttons()
        self.status = f"Manette : {self.device.name}"

    def _buttons(self):
        # Noms SDL : A = Croix (en bas), B = Rond (a droite), Y = Triangle (en haut).
        return (
            bool(self.device.get_button(pygame.CONTROLLER_BUTTON_A)),
            bool(self.device.get_button(pygame.CONTROLLER_BUTTON_B)),
            bool(self.device.get_button(pygame.CONTROLLER_BUTTON_Y))
        )

    def poll(self):
        if not self._ready:
            return GamepadInput()
        try:
            pygame.event.get()  # Actualiser les entrees et vider les evenements.
            if self.device is not None and not self.device.attached():
                self.device.quit()
                self.device = None
                self._previous_buttons = (False, False, False)
                self._next_scan = 0.0
                self.status = "Manette : deconnectee"
            if self.device is None:
                now = time.monotonic()
                if now >= self._next_scan:
                    self._next_scan = now + 1.0
                    self._scan()
                if self.device is None:
                    return GamepadInput()

            axes = (
                normalize_axis(self.device.get_axis(pygame.CONTROLLER_AXIS_LEFTX)),
                -normalize_axis(self.device.get_axis(pygame.CONTROLLER_AXIS_LEFTY)),
                -normalize_axis(self.device.get_axis(pygame.CONTROLLER_AXIS_RIGHTY)),
            )
            close, open_, reset = self._buttons()
            previous_close, previous_open, previous_reset = self._previous_buttons
            self._previous_buttons = (close, open_, reset)

            return GamepadInput(
                joint_axes=axes,
                close_gripper=close and not previous_close,
                open_gripper=open_ and not previous_open,
                reset_targets=reset and not previous_reset
            )
        
        except pygame.error as error:
            if self.device is not None:
                self.device.quit()
                self.device = None
            self.status = f"Manette : {error} (nouvel essai automatique)"
            self._next_scan = time.monotonic() + 1.0
            return GamepadInput()

    def close(self):
        if self.device is not None:
            self.device.quit()
            self.device = None
        if pygame is not None:
            controller.quit()
            pygame.display.quit()
        self._ready = False
