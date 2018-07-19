import pyglet
import yaml

MAPPING_FOLDER = 'mappings'  # TODO: move to configurable


class Controller:

    def __init__(self, env):
        self.enabled = True
        self.env = env
        self.mapping = None

    def _initialize(self):
        pyglet.clock.schedule_interval(self.update, 0.1)
        self.env.unwrapped.window.push_handlers(self)

    def load_mapping(self, controller_type, model):
        mapping_file = "{}/{}.{}.yaml".format(MAPPING_FOLDER, controller_type, model)
        with open(mapping_file) as mf:
            self.mapping = yaml.load(mf)

    def update(self, dt):
        if self.enabled:
            self._do_update(dt=dt)

    def _do_update(self, dt):
        raise NotImplementedError

    def on_button_pressed(self, button_key):
        button_action = self.mapping['buttons'][button_key]
        action = getattr(self, button_action)
        if action is not None:
            return action()

    def on_modifier_pressed(self, modifiers, action):
        modified_action = action

        for modifier in self.mapping['modifiers']:
            # if one of the mapped modifiers is active
            if modifiers[modifier]:
                # execute modifier over action
                modifier_method = getattr(self, self.mapping['modifiers'][modifier])
                if modifier_method is not None:
                    modified_action = modifier_method(modified_action)

        return modified_action

    # action
    def record(self):
        pass

    # action
    def step(self, action):
        response = self.env.step(action)
        self.env.render()
        return response

    # action
    def reset(self):
        self.env.reset()
        self.env.render()

    # modifier
    def boost(self, action):
        return action * self.mapping['config']['speed_boost']

