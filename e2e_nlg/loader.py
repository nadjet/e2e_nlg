from utils.data_load import *
from e2e_nlg.cleaner import *
from utils.log import logger

class E2ENLGDataLoader:
    def __init__(self, path, train_file, dev_file=None, percentile=95):
        self.data = None
        self.max_size = 0
        self.percentile = percentile
        self.has_dev = False
        self.train_ds = E2ENLGCleanedDataset(path,train_file)
        self.train_ds.clean()
        if dev_file is not None:
            self.dev_ds = E2ENLGCleanedDataset(path,dev_file)
            self.has_dev = True
            self.dev_ds.clean()

    def applyPercentile(self,src):
        p1 = np.percentile([len(o) for o in src.train.x.items] + [len(o) for o in src.valid.x.items], self.percentile)
        p2 = np.percentile([len(o) for o in src.train.y.items] + [len(o) for o in src.valid.y.items], self.percentile)
        if p1 > p2:
            max_size = p1
        else:
            max_size = p2
        max_size = int(max_size)
        max_size = max_size + 1
        src = src.filter_by_func(lambda x, y: len(x) > max_size or len(y) > max_size)
        return src, max_size

    def setDataAndMaxSize(self,path="."):
        if self.has_dev:
            self.train_ds.df["is_valid"] = False
            self.dev_ds.df["is_valid"] = True
            df = pd.concat([self.train_ds.df, self.dev_ds.df], ignore_index=True, axis=0)
        else:
            df = self.train_ds.df
        if "mr" in df.columns:
            df["MR"] = df["mr"]
            df.drop(["mr"], inplace=True, axis=1)
        if self.has_dev:
            src = Seq2SeqTextList.from_df(df, path=path, cols='MR').split_from_df(col='is_valid').label_from_df(cols='ref', label_cls=TextList)
            src, max_size = self.applyPercentile(src)
            self.max_size = max_size
            logger.info("Maximum size for inputs and outputs is: {}\n".format(self.max_size))

        else:
            src = Seq2SeqTextList.from_df(df, path=path, cols='MR').split_none().label_from_df(cols='ref', label_cls=TextList)
        self.data = src.databunch()

        logger.info("Size of input vocabulary={}".format(len(self.data.x.vocab.itos)))
        logger.info("Size of output vocabulary={}".format(len(self.data.y.vocab.itos)))
        logger.info(self.data)


    def save_data(self):
        self.data.save()

    def load_data(self, path="."):
        self.data = load_data(path)


