import numpy as np
import pyglet
from pyglet.input import DeviceOpenException

from controllers.devices.device_controller import DeviceController


class JoystickController(DeviceController):

    def __init__(self, env, device_id=0):
        DeviceController.__init__(self, env)
        self.device_id = device_id
        self.joystick = None
        self.x = 0.0
        self.y = 0.0

    def configure(self):
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
        self.joystick.push_handlers(self)
        # self.joystick.push_handlers(self.on_joyaxis_motion, self)
        # call general initialization routine
        DeviceController.configure(self)

    def on_joyaxis_motion(self, joystick, axis, value):
        if axis == 'x' and self.x != value:
            self.x = value
        if axis == 'y' and self.y != value:
            self.y = value

    def _do_update(self, dt):
        clean_x = round(self.y, 2)
        clean_z = round(self.x, 2)

        if clean_x == 0.0 and clean_z == 0.0:
            return None

        x = round(clean_x, 2)
        z = round(clean_z, 2)

        action = np.array([-x, -z])

        action = self.on_modifier_pressed(self.joystick.buttons, action)

        self.has_input = False

        return action

    def on_joybutton_press(self, joystick, button):
        self.on_button_pressed(button)

    def clean(self):
        self.x = 0.0
        self.y = 0.0
        print('Cleaning joystick spurious noise....')

