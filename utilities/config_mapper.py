import collections
from copy import deepcopy
import yaml


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_config(root, algorithm=None, env_name=None):
    config_dir = '{0}/{1}'
    config_dir2 = '{0}/{1}/{2}'
    default_name = 'default'
    with open(config_dir.format(root, "{}.yaml".format(default_name)), "r") as f:
        try:
            default_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    return default_config
