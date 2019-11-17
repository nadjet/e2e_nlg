import os

import fasttext as ft
import torch
from fastai.text import nn, tensor

from utils.log import logger


class Seq2SeqEmbeddings:

    def __init__(self, data, path, enc_emb='enc_emb.pth', dec_emb='dec_emb.pth'):
        self.data = data
        self.pretrained_embeddings = None
        self.emb_enc = None
        self.emb_dec = None
        self.enc_emb_path = os.path.join(path, enc_emb)
        self.dec_emb_path = os.path.join(path, dec_emb)

    def set_pretrained_embeddings(self, model_path='cc.en.300.bin'):
        self.pretrained_embeddings = ft.load_model(model_path)

    @staticmethod
    def build_embs(vecs, itos, em_sz=300):
        '''

        :param itos: word to index dictionary
        :param em_sz: embedding size
        :param mult:
        :return: embedding for each word in the vocabulary, and random for OOV
        '''
        emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
        wgts = emb.weight.data
        vec_dic = {w: vecs.get_word_vector(w) for w in vecs.get_words()}
        for i, w in enumerate(itos):
            try:
                wgts[i] = tensor(vec_dic[w])
            except:
                print(w)
                pass
        return emb

    def set_embeddings(self):
        self.emb_enc = Seq2SeqEmbeddings.build_embs(self.pretrained_embeddings, self.data.x.vocab.itos)
        self.emb_dec = Seq2SeqEmbeddings.build_embs(self.pretrained_embeddings, self.data.y.vocab.itos)

    def save_embeddings(self):
        torch.save(self.emb_enc, self.enc_emb_path)
        torch.save(self.emb_dec, self.dec_emb_path)
        logger.info("Encoder embeddings saved to: {}".format(self.enc_emb_path))
        logger.info("Encoder embeddings saved to: {}".format(self.dec_emb_path))


    def load_embeddings(self):
        self.emb_enc = torch.load(self.enc_emb_path)
        self.emb_dec = torch.load(self.dec_emb_path)
