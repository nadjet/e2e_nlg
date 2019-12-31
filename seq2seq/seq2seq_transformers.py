import math
from fastai.text import nn, ifnone, F, compose, Module
import torch
from utils.transformers import TransformersUtils


class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."

    def __init__(self, d):
        super().__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.) / d)))

    def forward(self, pos):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc

class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"

    def __init__(self, emb, inp_p=0.):
        super().__init__()
        self.emb_sz = 300
        self.embed = emb
        self.pos_enc = PositionalEncoding(300)
        self.drop = nn.Dropout(inp_p)

    def forward(self, inp):
        pos = torch.arange(0, inp.size(1), device=inp.device).float()
        return self.drop(self.embed(inp) * math.sqrt(self.emb_sz) + self.pos_enc(pos))

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head=None, p=0., bias=True, scale=True):
        super().__init__()
        d_head = ifnone(d_head, d_model // n_heads)
        self.n_heads, self.d_head, self.scale = n_heads, d_head, scale
        self.q_wgt, self.k_wgt, self.v_wgt = [nn.Linear(
            d_model, n_heads * d_head, bias=bias) for o in range(3)]
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att, self.drop_res = nn.Dropout(p), nn.Dropout(p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q, kv, mask=None):
        return self.ln(q + self.drop_res(self.out(self._apply_attention(q, kv, mask=mask))))

    def create_attn_mat(self, x, layer, bs):
        return layer(x).view(bs, x.size(1), self.n_heads, self.d_head
                             ).permute(0, 2, 1, 3)

    def _apply_attention(self, q, kv, mask=None):
        bs, seq_len = q.size(0), q.size(1)
        wq, wk, wv = map(lambda o: self.create_attn_mat(*o, bs),
                         zip((q, kv, kv), (self.q_wgt, self.k_wgt, self.v_wgt)))
        attn_score = wq @ wk.transpose(2, 3)
        if self.scale: attn_score /= math.sqrt(self.d_head)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = attn_prob @ wv
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)


class EncoderBlock(nn.Module):
    "Encoder block of a Transformer model."

    # Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff = TransformersUtils.feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)

    def forward(self, x, mask=None): return self.ff(self.mha(x, x, mask=mask))


class DecoderBlock(Module):
    "Decoder block of a Transformer model."

    # Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff = TransformersUtils.feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)

    def forward(self, x, enc, mask_out=None): return self.ff(self.mha2(self.mha1(x, x, mask_out), enc))


class Transformer(Module):
    def __init__(self, enc_size, dec_size, emb_enc, emb_dec, n_layers=6, n_heads=8, d_model=256, d_head=32,
                 d_inner=1024, p=0.1, bias=True, scale=True, double_drop=True, pad_idx=1):
        self.enc_emb = TransformerEmbedding(emb_enc, p)
        self.dec_emb = TransformerEmbedding(emb_dec, 0.)
        args = (n_heads, d_model, d_head, d_inner, p, bias, scale, double_drop)
        self.encoder = nn.ModuleList([EncoderBlock(*args) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(*args) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, dec_size)
        self.out.weight = self.dec_emb.embed.weight
        self.pad_idx = pad_idx

    def forward(self, inp, out):
        mask_out = TransformersUtils.get_output_mask(out, self.pad_idx)
        enc, out = self.enc_emb(inp), self.dec_emb(out)
        enc = compose(self.encoder)(enc)
        out = compose(self.decoder)(out, enc, mask_out)
        return self.out(out)