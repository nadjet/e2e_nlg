from utils.data_load import *
from e2e_nlg.loading.cleaner import *
from utils.log import logger
from utils.fastai_custom import TextList

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
        self.src = None
        self.max_size = 0

    def applyPercentile(self):
        p1 = np.percentile([len(o) for o in self.src.train.x.items] + [len(o) for o in self.src.valid.x.items], self.percentile)
        p2 = np.percentile([len(o) for o in self.src.train.y.items] + [len(o) for o in self.src.valid.y.items], self.percentile)
        if p1 > p2:
            max_size = p1
        else:
            max_size = p2
        max_size = int(max_size)
        max_size = max_size + 1
        self.src = self.src.filter_by_func(lambda x, y: len(x) > max_size or len(y) > max_size)
        self.max_size = max_size

    def setDataAndMaxSize(self,path=".",bs=64):
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
            self.src = Seq2SeqTextList.from_df(df, path=path, cols='MR').split_from_df(col='is_valid').label_from_df(cols='ref', label_cls=TextList)
            self.applyPercentile()
            logger.info("Maximum size for inputs and outputs is: {}\n".format(self.max_size))

        else:
            self.src = Seq2SeqTextList.from_df(df, path=path, cols='MR').split_none().label_from_df(cols='ref', label_cls=TextList)
        self.data = self.src.databunch(bs=bs)

        logger.info("Size of input vocabulary={}".format(len(self.data.x.vocab.itos)))
        logger.info("Size of output vocabulary={}".format(len(self.data.y.vocab.itos)))
        logger.info(self.data)


    def save_data(self):
        self.data.save()

    def load_data(self, path="."):
        self.data = load_data(path)

import sys
if __name__=="__main__":
    dl0 = E2ENLGDataLoader(sys.argv[1], "sample.csv", "sample.csv", percentile=100)
    dl0.setDataAndMaxSize()
    for i,row in dl0.train_ds.df.iterrows():
        print("\n",i)
        print(row["old_ref"])
        print(row["ref"])
        if i==10:
            break

