def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def identity(func):
    return func


profile = __builtins__.get('profile', identity)
