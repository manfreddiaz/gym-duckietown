import pyglet


class Controller:

    def __init__(self):
        self.enabled = True

    def _initialize(self):
        pyglet.clock.schedule_interval(self.update, 0.1, self)

    def update(self, dt):
        if self.enabled:
            self._do_update(dt=dt)

    def _do_update(self, dt):
        raise NotImplementedError

    def button_pressed(self):
        if self.enabled:
            self._do_button_pressed()

    def _do_button_pressed(self):
        raise NotImplementedError

    def record(self):
        pass


