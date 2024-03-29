import datetime
import json
import os


def load_state_dict(obj, state_dict, use_strict=True):
    if use_strict:
        try:
            obj.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(e)
            obj.load_state_dict(state_dict, strict=False)
    else:
        obj.load_state_dict(state_dict)


def version_of(ckpt_path, ascend=False):
    if ckpt_path is None:
        version = datetime.datetime.now().timestamp()
    else:
        ckpt_dir = os.path.dirname(ckpt_path)
        with open(os.path.join(ckpt_dir, 'meta.json'), 'r') as f:
            ckpt_path = json.load(f).get('ckpt_path')

        if ckpt_path is None or (not ascend):
            version = datetime.datetime.strptime(os.path.basename(ckpt_dir), '%Y-%m-%d-%H%M%S-%f').timestamp()
        else:
            version = version_of(ckpt_path)

    return version
