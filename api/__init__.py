class Mode:
    debug_label     = 'debug_label'
    auto_regress    = 'auto_regress'
    teacher_forcing = 'teacher_forcing'
    self_critical   = 'self_critical'

    __all__ = [debug_label, auto_regress, teacher_forcing, self_critical]
    label_modes = [auto_regress, teacher_forcing, self_critical]
    text_modes = [teacher_forcing, self_critical]

class Phase:
    train = 'train'
    val = 'val'
    test = 'test'


class Token:
    bos = '<bos>'
    eos = '<eos>'
    unk = '<unk>'
