class Mode:
    debug = 'debug'
    bootstrap = 'bootstrap'
    full = 'full'

    __all__ = [debug, bootstrap, full]


class Phase:
    train = 'train'
    val = 'val'
    test = 'test'


class Token:
    bos = '<bos>'
    eos = '<eos>'
    unk = '<unk>'
