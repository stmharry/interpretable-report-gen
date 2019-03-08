import enum


class Mode(enum.Flag):
    # dataset
    as_one_sentence = enum.auto()

    # model
    enc_image          = enum.auto()
    enc_with_attention = enum.auto()
    gen_label_all      = enum.auto()
    gen_label          = enum.auto()
    gen_text           = enum.auto()

    # loss
    use_label_all_ce    = enum.auto()
    use_label_ce        = enum.auto()
    use_stop_bce        = enum.auto()
    use_teacher_forcing = enum.auto()
    use_self_critical   = enum.auto()
    use_chexpert        = enum.auto()

    #
    base_uncond = as_one_sentence | gen_label | gen_text | use_stop_bce | use_teacher_forcing
    base_cond   = as_one_sentence | enc_image | gen_label | gen_text | use_stop_bce | use_teacher_forcing
    debug_label = enc_image | enc_with_attention | gen_label_all | use_label_all_ce
    pretrain    = enc_image | enc_with_attention | gen_label | use_label_ce | use_stop_bce
    full_tf     = enc_image | enc_with_attention | gen_label | gen_text | use_stop_bce | use_teacher_forcing
    full_sc     = enc_image | enc_with_attention | gen_label | gen_text | use_self_critical
    full        = enc_image | enc_with_attention | gen_label | gen_text | use_self_critical | use_chexpert


class Phase:
    train = 'train'
    val   = 'val'
    test  = 'test'


class Token:
    bos = '<bos>'
    eos = '<eos>'
    unk = '<unk>'
