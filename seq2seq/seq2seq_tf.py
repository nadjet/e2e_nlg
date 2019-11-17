from fastai.text import nn,one_param
import random
import torch

class Seq2SeqRNN_tf(nn.Module):
    def __init__(self, emb_enc, emb_dec, hidden_layer_size, max_output_length, nl=2, bos_idx=0, pad_idx=1):
        super().__init__()
        self.nl, self.hidden_layer_size, self.max_output_length = nl, hidden_layer_size, max_output_length
        self.bos_idx, self.pad_idx = bos_idx, pad_idx
        self.em_sz_enc = emb_enc.embedding_dim
        self.em_sz_dec = emb_dec.embedding_dim
        self.voc_sz_dec = emb_dec.num_embeddings

        self.emb_enc = emb_enc
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(self.em_sz_enc, hidden_layer_size, num_layers=nl,
                              dropout=0.25, batch_first=True)
        self.out_enc = nn.Linear(hidden_layer_size, self.em_sz_dec, bias=False)

        self.emb_dec = emb_dec
        self.gru_dec = nn.GRU(self.em_sz_dec, self.em_sz_dec, num_layers=nl,
                              dropout=0.1, batch_first=True)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(self.em_sz_dec, self.voc_sz_dec)
        self.out.weight.data = self.emb_dec.weight.data
        self.pr_force = 0.

    def encoder(self, bs, inp):
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        _, h = self.gru_enc(emb, h)
        h = self.out_enc(h)
        return h

    def decoder(self, dec_inp, h):
        emb = self.emb_dec(dec_inp).unsqueeze(1)
        outp, h = self.gru_dec(emb, h)
        outp = self.out(self.out_drop(outp[:, 0]))
        return h, outp

    def forward(self, inp, targ=None):
        bs, sl = inp.size()
        h = self.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx

        res = []
        for i in range(self.max_output_length):
            h, outp = self.decoder(dec_inp, h)
            res.append(outp)
            dec_inp = outp.max(1)[1]
            if (dec_inp == self.pad_idx).all(): break
            if (targ is not None) and (random.random() < self.pr_force):
                if i >= targ.shape[1]: continue
                dec_inp = targ[:, i]
        return torch.stack(res, dim=1)

    def initHidden(self, bs):
        return one_param(self).new_zeros(self.nl, bs, self.hidden_layer_size)

