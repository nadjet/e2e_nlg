from fastai.data_block import *
from fastai.data_block import ItemList
from fastai.text import *
from fastai.text.data import *

text_extensions = {'.txt'}


__all__ = ['ItemList', 'CategoryList', 'MultiCategoryList', 'MultiCategoryProcessor', 'LabelList', 'ItemLists', 'get_files',
           'PreProcessor', 'LabelLists', 'FloatList', 'CategoryProcessor', 'EmptyLabelList', 'MixedItem', 'MixedProcessor',
           'MixedItemList']



def _join_texts(texts:Collection[str], mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    bos_tok = f'{BOS} ' if include_bos else ''
    text_col = f'{bos_tok}{FLD} {1} ' + df[0].astype(str) if mark_fields else f'{bos_tok}' + df[0].astype(str)
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)
    if include_eos: text_col = text_col + f' {EOS}'
    return text_col.values

class TokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000,
                 mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields
        self.include_bos, self.include_eos = include_bos, include_eos

    def process_one(self, item):
        return self.tokenizer._process_all_1(_join_texts([item], self.mark_fields, self.include_bos, self.include_eos))[0]

    def process(self, ds):
        ds.items = _join_texts(ds.items, self.mark_fields, self.include_bos, self.include_eos)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

class NumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`."
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=3):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)
    def process(self, ds):
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)

class TextList(ItemList):
    "Basic `ItemList` for text data."
    _bunch = TextClasDataBunch
    _processor = [TokenizeProcessor, NumericalizeProcessor]
    _is_lm = False

    def __init__(self, items:Iterator, vocab:Vocab=None, pad_idx:int=1, sep=' ', **kwargs):
        super().__init__(items, **kwargs)
        self.vocab,self.pad_idx,self.sep = vocab,pad_idx,sep
        self.copy_new += ['vocab', 'pad_idx', 'sep']

    def get(self, i):
        o = super().get(i)
        return o if self.vocab is None else Text(o, self.vocab.textify(o, self.sep))

    def label_for_lm(self, **kwargs):
        "A special labelling method for language models."
        self.__class__ = LMTextList
        kwargs['label_cls'] = LMLabelList
        return self.label_const(0, **kwargs)

    def reconstruct(self, t:Tensor):
        idx_min = (t != self.pad_idx).nonzero().min()
        idx_max = (t != self.pad_idx).nonzero().max()
        return Text(t[idx_min:idx_max+1], self.vocab.textify(t[idx_min:idx_max+1]))

    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=text_extensions, vocab:Vocab=None,
                    processor:PreProcessor=None, **kwargs)->'TextList':
        "Get the list of files in `path` that have a text suffix. `recurse` determines if we search subfolders."
        processor = ifnone(processor, [OpenFileProcessor(), TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        return super().from_folder(path=path, extensions=extensions, processor=processor, **kwargs)

    def show_xys(self, xs, ys, max_len:int=70)->None:
        "Show the `xs` (inputs) and `ys` (targets). `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        names = ['idx','text'] if self._is_lm else ['text','target']
        items = []
        for i, (x,y) in enumerate(zip(xs,ys)):
            txt_x = ' '.join(x.text.split(' ')[:max_len]) if max_len is not None else x.text
            items.append([i, txt_x] if self._is_lm else [txt_x, y])
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))

    def show_xyzs(self, xs, ys, zs, max_len:int=70):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions). `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        items,names = [],['text','target','prediction']
        for i, (x,y,z) in enumerate(zip(xs,ys,zs)):
            txt_x = ' '.join(x.text.split(' ')[:max_len]) if max_len is not None else x.text
            items.append([txt_x, y, z])
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))
