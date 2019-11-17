import torch
from fastai.text import CrossEntropyFlat,F

class Loss:

    @staticmethod
    def seq2seq_loss(out, targ, pad_idx=1):
        batch_size, targ_len = targ.size()
        _, out_len, vs = out.size()
        if targ_len > out_len: out = F.pad(out, (0, 0, 0, targ_len - out_len, 0, 0), value=pad_idx)
        if out_len > targ_len: targ = F.pad(targ, (0, out_len - targ_len, 0, 0), value=pad_idx)
        return CrossEntropyFlat()(out, targ)