from fastai.text import F

class Metrics:

    @staticmethod
    def seq2seq_acc(out, targ, pad_idx=1):
        bs,targ_len = targ.size()
        _,out_len,vs = out.size()
        if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
        if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
        out = out.argmax(2)
        return (out==targ).float().mean()