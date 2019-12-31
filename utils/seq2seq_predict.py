import torch
from fastai.text import F, DatasetType, progress_bar
from random import choice


class PredictUtils:
    def __init__(self, learn):
        self.learn = learn

    # softmax = e^(z/T) / sum_i e^(z_i/T)
    @staticmethod
    def softmax_calibrated(x, scale=1.):
        if x.ndim == 1:
            x = x.reshape((1, -1))

        max_x = x.max(dim=1)[0].reshape((-1, 1))
        exp_x = torch.exp((x - max_x) / scale)

        return exp_x / exp_x.sum(dim=1).reshape((-1, 1))

    @staticmethod
    def select_nucleus(outp, p=0.5, T=1):
        probs = PredictUtils.softmax_calibrated(outp, T)
        probs = probs[0]
        #probs = F.softmax(outp,dim=-1)
        idxs = torch.argsort(probs, descending=True)
        res,cumsum = [],0.
        for idx in idxs:
            res.append(idx)
            cumsum += probs[idx]
            # if probability greater then p, then we select output, so if one output has probability of 0.9 and that's the value of p, then only that one get selected, or 2 if 0.5 and 0.4, or 9 of 0.1
            # so the lower the probability, the less outputs are selected
            if cumsum>p:
                index = choice(range(len(res)))
                return idxs.new_tensor([res[index]]), probs[index]

    @staticmethod
    def select_topk(outp, k=5):
        probs = F.softmax(outp,dim=-1)
        vals,idxs = probs.topk(k, dim=-1)
        return idxs[torch.randint(k, (1,))]


    '''
    def decode(model, inp,p=0.5):
        inp = inp[None]
        bs, sl = inp.size()
        hid,enc_out = model.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + model.bos_idx
        enc_att = model.enc_att(enc_out)
    
        res = []
        for i in range(model.out_sl):
            hid, outp = model.decoder(dec_inp, hid, enc_att, enc_out)
            #dec_inp = select_nucleus(outp[0], p)
            dec_inp = select_topk(outp[0], k=1)
            res.append(dec_inp)
            if (dec_inp==model.pad_idx).all(): break
        return torch.cat(res)
    '''


    def decode(self, inp, p, T):
        inp = inp[None]
        bs, sl = inp.size()
        hid = self.learn.model.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + self.learn.model.bos_idx
    
        res = []
        for i in range(self.learn.model.max_output_length):
            hid, outp = self.learn.model.decoder(dec_inp, hid)
            dec_inp,_ = PredictUtils.select_nucleus(outp[0], p, T)
            #dec_inp = select_topk(outp[0], k=5)
            res.append(dec_inp)
            if (dec_inp==self.learn.model.pad_idx).all(): break
        return torch.cat(res)


    def decode_top_p(self, inp, p, T=1.):
        inp = inp[None]
        bs, sl = inp.size()
        hid = self.learn.model.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + self.learn.model.bos_idx
        result = []
        probs = []
        for i in range(self.learn.model.max_output_length):
            hid, outp = self.learn.model.decoder(dec_inp, hid)
            dec_inp, prob = PredictUtils.select_nucleus(outp[0], p, T)
            if (dec_inp == self.learn.model.pad_idx).all():
                break
            result.append(dec_inp)
            probs.append(prob)
        return probs, torch.cat(result)

    @staticmethod
    def calculate_precision(trues, predictions):
        num_correct = len(set.intersection(set(trues),set(predictions)))
        pr = 0
        if len(set(predictions))>0:
            pr = float(num_correct)/len(set(predictions))
        return pr

    @staticmethod
    def calculate_recall(trues, predictions):
        num_correct = len(set.intersection(set(trues),set(predictions)))
        rec = 0
        if len(set(trues))>0:
            rec = float(num_correct) / len(set(trues))
        return rec

    @staticmethod
    def calculate_fscore(trues, predictions,beta=0.5):
        num_correct = len(set.intersection(set(trues),set(predictions)))
        rec = 0
        pr = 0
        if len(set(predictions))>0:
            pr = float(num_correct)/len(set(predictions))
        if len(set(trues))>0:
            rec = float(num_correct) / len(set(trues))
        if pr+rec >0:
            f_score = ((1+beta**2) * (pr * rec))/(((beta**2)*pr)+rec)
        else:
            f_score = 0.
        return f_score

    def predict_with_decode(self, x, y, p=0.5, T=1.):
        self.learn.model.eval()
        ds = self.learn.data.train_ds
        with torch.no_grad():
            probs, out = self.decode_top_p(x,p, T)
            rx = ds.x.reconstruct(x)
            ry = ds.y.reconstruct(y)
            rz = ds.y.reconstruct(out)
            return rx,ry,rz



    def get_predictions_transformer(self, ds_type=DatasetType.Valid):
        self.learn.model.eval()
        inputs, targets, outputs = [],[],[]
        with torch.no_grad():
            for xb,yb in progress_bar(self.learn.dl(ds_type)):
                out = self.learn.model(*xb)
                for x,y,z in zip(xb[0],xb[1],out):
                    inputs.append(self.learn.data.train_ds.x.reconstruct(x))
                    targets.append(self.learn.data.train_ds.y.reconstruct(y))
                    outputs.append(self.learn.data.train_ds.y.reconstruct(z.argmax(1)))
        return inputs, targets, outputs

    def get_predictions(self, ds_type=DatasetType.Valid):
        self.learn.model.eval()
        inputs, targets, outputs = [], [], []
        with torch.no_grad():
            for xb, yb in progress_bar(self.learn.dl(ds_type)):
                out = self.learn.model(xb)
                for x, y, z in zip(xb, yb, out):
                    inputs.append(self.learn.data.train_ds.x.reconstruct(x))
                    targets.append(self.learn.data.train_ds.y.reconstruct(y))
                    outputs.append(self.learn.data.train_ds.y.reconstruct(z.argmax(1)))
        return inputs, targets, outputs


    def preds_acts(self, ds_type=DatasetType.Valid):
        "Same as `get_predictions` but also returns non-reconstructed activations"
        self.learn.model.eval()
        ds = self.learn.data.train_ds
        rxs,rys,rzs,xs,ys,zs = [],[],[],[],[],[] # 'r' == 'reconstructed'
        with torch.no_grad():
            for xb,yb in progress_bar(self.learn.dl(ds_type)):
                out = self.learn.model(xb)
                for x,y,z in zip(xb,yb,out):
                    rxs.append(ds.x.reconstruct(x))
                    rys.append(ds.y.reconstruct(y))
                    rzs.append(ds.y.reconstruct(z.argmax(1)))
                    for a,b in zip([xs,ys,zs],[x,y,z]): a.append(b)
        return rxs,rys,rzs,xs,ys,zs

