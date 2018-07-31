from controllers import Controller
from replay import RecordingController


class ImitationLearningRecorder(RecordingController):

    def __init__(self, env, iil_controller, stop_on_done=True, stop_on_reset=True):
        RecordingController.__init__(self, env, iil_controller, stop_on_done, stop_on_reset)

    def configure(self):
        capabilities = {
            'record': self.record,
            'stop': self.stop
        }
        Controller.extend_capabilities(self, self._controller.primary, capabilities)
        self._controller.configure()

    def reset(self, **kwargs):
        self._controller.reset(**kwargs)
