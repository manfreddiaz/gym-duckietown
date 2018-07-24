import yaml

from controllers.base_controller import Controller


class DeviceController(Controller):

    def __init__(self, env):
        Controller.__init__(self, env)

    def load_mapping(self, path):
        with open(path) as mf:
            self.mapping = yaml.load(mf)

    def _do_update(self, dt):
        raise NotImplementedError

    def on_button_pressed(self, button_key):
        button_action = self.mapping['buttons'][button_key]
        action = getattr(self, button_action)
        if action is not None:
            return action()

    def on_modifier_pressed(self, modifiers, action):
        modified_action = action

        if 'modifiers' in self.mapping:
            for modifier in self.mapping['modifiers']:
                # if one of the mapped modifiers is active
                if modifiers[modifier]:
                    # execute modifier over input action
                    modifier_method = getattr(self, self.mapping['modifiers'][modifier])
                    if modifier_method is not None:
                        modified_action = modifier_method(modified_action)

        return modified_action

    # modifier
    def boost(self, action):
        return action * self.mapping['config']['speed_boost']
