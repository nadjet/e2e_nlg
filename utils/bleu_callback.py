from fastai.text import Callback,Counter
import numpy as np
from math import exp
from fastai.text import add_metrics


class NGram():
    def __init__(self, ngram, max_n=5000): self.ngram, self.max_n = ngram, max_n

    def __eq__(self, other):
        if len(self.ngram) != len(other.ngram): return False
        return np.all(np.array(self.ngram) == np.array(other.ngram))

    def __hash__(self): return int(sum([o * self.max_n ** i for i, o in enumerate(self.ngram)]))


class CorpusBLEU(Callback):
    def __init__(self, vocab_sz):
        self.vocab_sz = vocab_sz
        self.name = 'bleu'

    def on_epoch_begin(self, **kwargs):
        self.pred_len, self.targ_len, self.corrects, self.counts = 0, 0, [0] * 4, [0] * 4

    @staticmethod
    def get_grams(x, n, max_n=5000):
        return x if n == 1 else [NGram(x[i:i + n], max_n=max_n) for i in range(len(x) - n + 1)]

    @staticmethod
    def get_correct_ngrams(pred, targ, n, max_n=5000):
        pred_grams, targ_grams = CorpusBLEU.get_grams(pred, n, max_n=max_n), CorpusBLEU.get_grams(targ, n, max_n=max_n)
        pred_cnt, targ_cnt = Counter(pred_grams), Counter(targ_grams)
        return sum([min(c, targ_cnt[g]) for g, c in pred_cnt.items()]), len(pred_grams)

    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output.argmax(dim=-1)
        for pred, targ in zip(last_output.cpu().numpy(), last_target.cpu().numpy()):
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            for i in range(4):
                c, t = CorpusBLEU.get_correct_ngrams(pred, targ, i + 1, max_n=self.vocab_sz)
                self.corrects[i] += c
                self.counts[i] += t

    def on_epoch_end(self, last_metrics, **kwargs):
        precs = [c / t for c, t in zip(self.corrects, self.counts)]
        len_penalty = exp(1 - self.targ_len / self.pred_len) if self.pred_len < self.targ_len else 1
        bleu = len_penalty * ((precs[0] * precs[1] * precs[2] * precs[3]) ** 0.25)
        return add_metrics(last_metrics, bleu)
