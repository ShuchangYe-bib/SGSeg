import copy
import os
from ast import literal_eval
import yaml

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree.
    It's a simple dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if isinstance(v, dict):
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            separator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), separator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())

def load_cfg_from_cfg_file(file):
    """
    Load configuration from a yaml file and convert it to a CfgNode.
    """
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg

def merge_cfg_from_list(cfg, cfg_list):
    """
    Merge configuration from a list into an existing configuration.
    """
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(value, cfg[subkey], subkey, full_key)
        setattr(new_cfg, subkey, value)

    return new_cfg

def _decode_cfg_value(v):
    """
    Decodes a raw config value (e.g., from a yaml config files or command line argument)
    into a Python object.
    """
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    return v

def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """
    Checks that `replacement`, which is intended to replace `original`, is of the right type.
    The type is correct if it matches exactly or is one of a few cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for from_type, to_type in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


