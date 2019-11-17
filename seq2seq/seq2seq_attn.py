import torch
from fastai.text import nn,F,one_param
import random
import math

class Seq2SeqRNN_attn(nn.Module):
    def __init__(self, emb_enc, emb_dec, nh, out_sl, nl=2, bos_idx=0, pad_idx=1):
        super().__init__()
        self.nl, self.nh, self.out_sl, self.pr_force = nl, nh, out_sl, 1
        self.bos_idx, self.pad_idx = bos_idx, pad_idx
        self.emb_enc, self.emb_dec = emb_enc, emb_dec
        self.emb_sz_enc, self.emb_sz_dec = emb_enc.embedding_dim, emb_dec.embedding_dim
        self.voc_sz_dec = emb_dec.num_embeddings

        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(self.emb_sz_enc, nh, num_layers=nl, dropout=0.25,
                              batch_first=True, bidirectional=True)
        self.out_enc = nn.Linear(2 * nh, self.emb_sz_dec, bias=False)

        self.gru_dec = nn.GRU(self.emb_sz_dec + 2 * nh, self.emb_sz_dec, num_layers=nl,
                              dropout=0.1, batch_first=True)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(self.emb_sz_dec, self.voc_sz_dec)
        self.out.weight.data = self.emb_dec.weight.data

        self.enc_att = nn.Linear(2 * nh, self.emb_sz_dec, bias=False)
        self.hid_att = nn.Linear(self.emb_sz_dec, self.emb_sz_dec)
        self.V = self.init_param(self.emb_sz_dec)

    def encoder(self, bs, inp):
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, hid = self.gru_enc(emb, 2 * h)

        pre_hid = hid.view(2, self.nl, bs, self.nh).permute(1, 2, 0, 3).contiguous()
        pre_hid = pre_hid.view(self.nl, bs, 2 * self.nh)
        hid = self.out_enc(pre_hid)
        return hid, enc_out

    def decoder(self, dec_inp, hid, enc_att, enc_out):
        hid_att = self.hid_att(hid[-1])
        # we have put enc_out and hid through linear layers
        u = torch.tanh(enc_att + hid_att[:, None])
        # we want to learn the importance of each time step
        attn_wgts = F.softmax(u @ self.V, 1)
        # weighted average of enc_out (which is the output at every time step)
        ctx = (attn_wgts[..., None] * enc_out).sum(1)
        emb = self.emb_dec(dec_inp)
        # concatenate decoder embedding with context (we could have just
        # used the hidden state that came out of the decoder, if we weren't
        # using attention)
        outp, hid = self.gru_dec(torch.cat([emb, ctx], 1)[:, None], hid)
        outp = self.out(self.out_drop(outp[:, 0]))
        return hid, outp

    def show(self, nm, v):
        if False: print(f"{nm}={v[nm].shape}")

    def forward(self, inp, targ=None):
        bs, sl = inp.size()
        hid, enc_out = self.encoder(bs, inp)
        #        self.show("hid",vars())
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx
        enc_att = self.enc_att(enc_out)

        res = []
        for i in range(self.out_sl):
            hid, outp = self.decoder(dec_inp, hid, enc_att, enc_out)
            res.append(outp)
            dec_inp = outp.max(1)[1]
            if (dec_inp == self.pad_idx).all(): break
            if (targ is not None) and (random.random() < self.pr_force):
                if i >= targ.shape[1]: continue
                dec_inp = targ[:, i]
        return torch.stack(res, dim=1)

    def initHidden(self, bs):
        return one_param(self).new_zeros(2 * self.nl, bs, self.nh)

    def init_param(self, *sz):
        return nn.Parameter(torch.randn(sz) / math.sqrt(sz[0]))