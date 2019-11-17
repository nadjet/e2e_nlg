import torch
from fastai.text import F, DatasetType, progress_bar
from random import choice



def select_nucleus(outp, p=0.5):
    probs = F.softmax(outp,dim=-1)
    idxs = torch.argsort(probs, descending=True)
    res,cumsum = [],0.
    for idx in idxs:
        res.append(idx)
        cumsum += probs[idx]
        # if probability greater then p, then we select output, so if one output has probability of 0.9 and that's the value of p, then only that one get selected, or 2 of 0.5 and 0.4, or 9 of 0.1
        # so the lower the probability, the less outputs are selected
        if cumsum>p: return idxs.new_tensor([choice(res)])

def select_topk(outp, k=5):
    probs = F.softmax(outp,dim=-1)
    vals,idxs = probs.topk(k, dim=-1)
    return idxs[torch.randint(k, (1,))]

'''
def decode(model, inp):
    inp = inp[None]
    bs, sl = inp.size()
    hid,enc_out = model.encoder(bs, inp)
    dec_inp = inp.new_zeros(bs).long() + model.bos_idx
    enc_att = model.enc_att(enc_out)

    res = []
    for i in range(model.out_sl):
        hid, outp = model.decoder(dec_inp, hid, enc_att, enc_out)
        #dec_inp = select_nucleus(outp[0], p=0.5)
        dec_inp = select_topk(outp[0], k=1)
        res.append(dec_inp)
        if (dec_inp==model.pad_idx).all(): break
    return torch.cat(res)
'''

def decode(model, inp):
    inp = inp[None]
    bs, sl = inp.size()
    hid = model.encoder(bs, inp)
    dec_inp = inp.new_zeros(bs).long() + model.bos_idx

    res = []
    for i in range(model.max_output_length):
        hid, outp = model.decoder(dec_inp, hid)
        #dec_inp = select_nucleus(outp[0], p=0.5)
        dec_inp = select_topk(outp[0], k=1)
        res.append(dec_inp)
        if (dec_inp==model.pad_idx).all(): break
    return torch.cat(res)

def predict_with_decode(learn, x, y):
    learn.model.eval()
    ds = learn.data.train_ds
    with torch.no_grad():
        out = decode(learn.model, x)
        rx = ds.x.reconstruct(x)
        ry = ds.y.reconstruct(y)
        rz = ds.y.reconstruct(out)
    return rx,ry,rz

def get_predictions(learn, ds_type=DatasetType.Valid):
    learn.model.eval()
    inputs, targets, outputs = [],[],[]
    with torch.no_grad():
        for xb,yb in progress_bar(learn.dl(ds_type)):
            out = learn.model(xb)
            for x,y,z in zip(xb,yb,out):
                inputs.append(learn.data.train_ds.x.reconstruct(x))
                targets.append(learn.data.train_ds.y.reconstruct(y))
                outputs.append(learn.data.train_ds.y.reconstruct(z.argmax(1)))
    return inputs, targets, outputs


def preds_acts(learn, ds_type=DatasetType.Valid):
    "Same as `get_predictions` but also returns non-reconstructed activations"
    learn.model.eval()
    ds = learn.data.train_ds
    rxs,rys,rzs,xs,ys,zs = [],[],[],[],[],[] # 'r' == 'reconstructed'
    with torch.no_grad():
        for xb,yb in progress_bar(learn.dl(ds_type)):
            out = learn.model(xb)
            for x,y,z in zip(xb,yb,out):
                rxs.append(ds.x.reconstruct(x))
                rys.append(ds.y.reconstruct(y))
                preds = z.argmax(1)
                rzs.append(ds.y.reconstruct(preds))
                for a,b in zip([xs,ys,zs],[x,y,z]): a.append(b)
    return rxs,rys,rzs,xs,ys,zs