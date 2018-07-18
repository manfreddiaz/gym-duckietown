import pyglet
from pyglet.input import DeviceOpenException

from controllers.base_controller import Controller


class JoystickController(Controller):

    def __init__(self, device_id=0):
        Controller.__init__(self)
        self.device_id = device_id
        self._initialize()

    def _initialize(self):
        Controller._initialize(self)
        # enumerate all available joysticks and select the one with id = device_id
        joysticks = pyglet.input.get_joysticks()
        if not joysticks:
            raise ConnectionError('No joysticks found on this device.')
        # check if device_id is valid
        if len(joysticks) >= self.device_id:
            raise ConnectionError('No joystick with id = {}.'.format(self.device_id))
        # select the joystick
        self.joystick = joysticks[self.device_id]
        # try to open
        try:
            self.joystick.open()
        except DeviceOpenException:
            raise ConnectionError('Joystick with id = {} is already in use.'.format(self.device_id))
        # register this controller as a handler
        self.joystick.push_handlers(self.button_pressed, self)

    def _do_update(self, dt):
        pass

    def _do_button_pressed(self):
        pass

