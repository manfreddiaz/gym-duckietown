from imitation.teachers.ua_duckietown_pipeline import UADuckietownPipelinePolicy
from imitation.training._duckiebot_env import DuckiebotEnvIcra2019


def robot(host='hero.local'):
    env = DuckiebotEnvIcra2019(host=host)
    env.observe()
    return env


def robot_teacher(env):
    return UADuckietownPipelinePolicy(env=env)

