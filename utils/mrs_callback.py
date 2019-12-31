from fastai.text import Callback
from fastai.text import add_metrics
import re

class CorpusMrs(Callback):
    def __init__(self, index_to_string_array, classifier):
        self.index_to_string_array = index_to_string_array
        self.classifier = classifier
        self.name = 'mrs'

    def textify(self, vec):
        str = []
        maj = False
        up = False
        for i in vec:
            print(i)
            word = self.index_to_string_array[i]
            if word=="xxup":
                up = True
                continue
            elif word=="xxmaj":
                maj = True
                continue
            elif word=="xxbos" or word=="xxpad":
                continue
            elif maj:
                str.append(word[0].upper()+word[1:])
                maj = False
            elif up:
                str.append(word.upper())
                up = False
            else:
                str.append(word)
        str = " ".join(str)
        str = re.sub(r" +([\.\-,])",r"\1",str)
        str = re.sub(r"  +"," ",str)
        return str

    def on_epoch_begin(self, **kwargs):
        self.pred_len, self.targ_len, self.corrects, self.counts = 0, 0, 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output.argmax(dim=-1)
        for pred, targ in zip(last_output.cpu().numpy(), last_target.cpu().numpy()):
            pred_str = self.textify(pred.tolist())
            targ_str = self.textify(targ.tolist())
            pred_mrs = self.classifier.predict(pred_str)[0]
            targ_mrs = self.classifier.predict(targ_str)[0]
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            if pred_mrs == targ_mrs:
                self.corrects += 1
            self.counts += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        precs_mrs = self.corrects / float(self.counts)
        return add_metrics(last_metrics, precs_mrs)
