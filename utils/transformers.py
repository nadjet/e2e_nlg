import torch
from fastai.text import nn,SequentialEx,MergeLayer, F

class TransformersUtils :

    @staticmethod
    def get_output_mask(inp, pad_idx=1):
        return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].bool()
    #     return ((inp == pad_idx)[:,None,:,None].long() + torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None] != 0)

    @staticmethod
    def shift_tfm(b):
        x,y = b
        y = F.pad(y, (1, 0), value=1)
        return [x,y[:,:-1]], y[:,1:]

    @staticmethod
    def feed_forward(d_model, d_ff, ff_p=0., double_drop=True):
        layers = [nn.Linear(d_model, d_ff), nn.ReLU()]
        if double_drop: layers.append(nn.Dropout(ff_p))
        return SequentialEx(*layers, nn.Linear(d_ff, d_model), nn.Dropout(ff_p), MergeLayer(), nn.LayerNorm(d_model))