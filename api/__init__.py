import enum

class Mode(enum.Flag):
    gen_label_all       = enum.auto()
    gen_label           = enum.auto()
    gen_text            = enum.auto()

    use_label_all_ce    = enum.auto()
    use_label_ce        = enum.auto()
    use_stop_bce        = enum.auto()
    use_teacher_forcing = enum.auto()
    use_self_critical   = enum.auto()
    use_chexpert        = enum.auto()

    debug_label = gen_label_all | use_label_all_ce
    pretrain    = gen_label | use_label_ce | use_stop_bce
    debug_text  = gen_label | gen_text | use_stop_bce | use_teacher_forcing
    full_tf     = gen_label | gen_text | use_label_ce | use_stop_bce | use_teacher_forcing
    full_sc     = gen_label | gen_text | use_label_ce | use_stop_bce | use_self_critical


class Phase:
    train = 'train'
    val = 'val'
    test = 'test'


class Token:
    bos = '<bos>'
    eos = '<eos>'
    unk = '<unk>'
