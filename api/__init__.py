class Mode:
    debug_label     = 'debug_label'
    pretrain        = 'pretrain'
    teacher_forcing = 'teacher_forcing'
    self_critical   = 'self_critical'

    __all__ = [debug_label, pretrain, teacher_forcing, self_critical]
    label_modes = [pretrain, teacher_forcing, self_critical]
    text_modes = [teacher_forcing, self_critical]

class Phase:
    train = 'train'
    val = 'val'
    test = 'test'


class Token:
    bos = '<bos>'
    eos = '<eos>'
    unk = '<unk>'
