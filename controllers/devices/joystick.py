import numpy as np
import pyglet
from pyglet.input import DeviceOpenException

from controllers.devices.device_controller import DeviceController


# Adapted from Bhairav Mehta initial implementation.
class JoystickController(DeviceController):

    def __init__(self, env, device_id=0):
        self.device_id = device_id
        self.joystick = None
        DeviceController.__init__(self, env)

    def _initialize(self):
        # enumerate all available joysticks and select the one with id = device_id
        joysticks = pyglet.input.get_joysticks()
        if not joysticks:
            raise ConnectionError('No joysticks found on this computer.')
        # check if device_id is valid
        if len(joysticks) <= self.device_id:
            raise ConnectionError('No joystick with id = {} found.'.format(self.device_id))
        # select the joystick
        self.joystick = joysticks[self.device_id]
        # try to open
        try:
            self.joystick.open()
        except DeviceOpenException:
            raise ConnectionError('Joystick with id = {} is already in use.'.format(self.device_id))
        # register this controller as a handler
        self.joystick.push_handlers(self.on_joybutton_press, self)
        # call general initialization routine
        DeviceController._initialize(self)

    def _do_update(self, dt):
        if round(self.joystick.x, 2) == 0.0 and round(self.joystick.y, 2) == 0.0:
            return

        x = round(self.joystick.y, 2)
        z = round(self.joystick.x, 2)

        action = np.array([-x, -z])

        action = self.on_modifier_pressed(self.joystick.buttons, action)

        self.step(action=action)

    def on_joybutton_press(self, joystick, button):
        self.on_button_pressed(button)


