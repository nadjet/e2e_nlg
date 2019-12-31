from fastai.text import *
import pandas as pd
import unicodedata
import re
from utils.log import logger

class MrPredictor:

    def __init__(self, main_path, dataset_path,train_file="trainset.csv", dev_file="devset.csv", test_file="testset_w_refs.csv"):

        self.main_path = main_path
        self.train_df = pd.read_csv(os.path.join(dataset_path,train_file),sep=",")
        self.train_df = MrPredictor.process_features(self.train_df)
        self.dev_df = pd.read_csv(os.path.join(dataset_path,dev_file),sep=",")
        self.dev_df = MrPredictor.process_features(self.dev_df)
        self.test_df = pd.read_csv(os.path.join(dataset_path,test_file), sep=",")
        self.test_df = MrPredictor.process_features(self.test_df)
        self.df_all = pd.concat([self.train_df, self.dev_df, self.test_df], ignore_index=True)
        print(self.df_all.shape)
        with open(os.path.join(os.path.join(main_path, "models"), 'vocab.pkl'), 'rb') as fp:
            vocab = pickle.load(fp)
        self.set_data_class(vocab,bs=56)
        self.load_model()

    @staticmethod
    def delexicalize(value, new_value, text):
        text = re.sub(value, new_value, text)
        text = re.sub(value.lower(), new_value.lower(), text)
        text = re.sub(MrPredictor.strip_accents(value.lower()), new_value.lower(), text)
        text = re.sub(MrPredictor.strip_accents(value), new_value, text)
        value0 = value[0] + value[1:].lower()
        text = re.sub(value0, new_value, text)
        text = re.sub(MrPredictor.strip_accents(value0), new_value, text)
        value0 = value[0].lower() + value[1:]
        text = re.sub(value0, new_value, text)
        text = re.sub(MrPredictor.strip_accents(value0), new_value, text)
        return text

    @staticmethod
    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    @staticmethod
    def process_row(row):
        row["old_ref"] = row["ref"]
        row["old_mr"] = row["mr"]
        row["ref"] = re.sub("  +", " ", row["ref"])
        row["mr"] = re.sub("  +", " ", row["mr"])
        name = re.sub(r"^.*name\[([^\]]+)\].*$", r"\1", row["mr"].strip())
        near = re.sub(r"^.*near\[([^\]]+)\].*$", r"\1", row["mr"].strip())
        name = re.sub("  +", " ", name)
        near = re.sub("  +", " ", near)
        row["ref"] = MrPredictor.delexicalize(name, "Xxx", row["ref"])
        row["ref"] = MrPredictor.delexicalize(near, "Yyy", row["ref"])
        row["mr"] = re.sub(r"name\[[^\]]+\](, *| *$)", "", row["mr"].strip())
        row["mr"] = re.sub(r"near\[[^\]]+\](, *| *$)", r"near[yes]\1", row["mr"].strip())
        row["mr"] = re.sub(r", *$", "", row["mr"].strip())
        row["mr"] = re.sub(r" *, *", ",", row["mr"].strip())
        row["mr"] = row["mr"].strip()
        return row

    @staticmethod
    def process_features(df):
        rows = []
        for i, row in df.iterrows():
            row0 = row.to_dict()
            row0 = MrPredictor.process_row(row0)
            if row["ref"] == row0["ref"]:
                continue
            rows.append(row0)
        return pd.DataFrame(rows)

    def set_data_class(self,vocab,bs=56):
        self.data_clas = TextClasDataBunch.from_df(".", train_df=self.train_df, valid_df=self.dev_df,
                                              vocab=vocab,
                                              text_cols='ref',
                                              label_cols='mr',
                                              label_delim=',',
                                              bs=bs)

    def load_model(self):
        assert(self.data_clas is not None)
        self.learn = text_classifier_learner(self.data_clas, arch=AWD_LSTM, drop_mult=1e-7)
        self.learn.path = Path(self.main_path)
        self.learn = self.learn.load("classifier_model")

    def predict(self,words):
        return self.learn.predict(words)

    def predict_in_out(self,in_file,out_file):
        df = pd.read_csv(in_file)
        df = MrPredictor.process_features(df)
        df["diff"] = ""
        df["ratio_diff_with_input"] = 0.
        df["num_diff"] = 0
        df["pred"] = ""
        import csv
        with open(out_file, "w") as f:
            w = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            w.writerow(["mr", "ref", "prediction", "diff_with_input", "ratio_diff_with_input", "num_diff_input",
                        "diff_with_pred", "ratio_diff_with_pred", "num_diff_pred"])
            for i, row in df.iterrows():
                sentence = row["ref"]
                prediction = self.predict(sentence)
                prediction = set([item.strip() for item in str(prediction[0]).split(";")])
                input = set([item.strip() for item in row["mr"].split(",")])
                new_input = set()
                for item in input:
                    if item.startswith("near"):
                        new_input.add("near[yes]")
                    elif item.startswith("name"):
                        continue
                    else:
                        new_input.add(item)
                input = new_input
                diff1 = input.difference(prediction)
                diff2 = prediction.difference(input)
                row["diff1"] = ",".join(sorted(list(diff1)))
                row["diff2"] = ",".join(sorted(list(diff2)))
                row["diff_with_input"] = ",".join(list(diff1))
                row["num_diff1"] = len(diff1)
                row["ratio_diff_with_input"] = len(diff1) / float(len(input))
                row["diff_with_pred"] = ",".join(list(diff2))
                row["num_diff2"] = len(diff2)
                row["ratio_diff_with_pred"] = len(diff2) / float(len(input))
                prediction = ",".join(sorted(list(prediction)))
                w.writerow([row['old_mr'], row['old_ref'], prediction, row['diff1'], row['ratio_diff_with_input'],
                            row['num_diff1'], row['diff2'], row['ratio_diff_with_pred'], row['num_diff2']])
                print(i)




import sys
if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    out_folder = sys.argv[3]
    train_file = "trainset.csv"
    dev_file = "devset.csv"
    test_file = "testset_w_refs.csv"
    logger.info("Loading classifier model...")
    predictor = MrPredictor(model_path, dataset_path, train_file, dev_file, test_file)
    logger.info("...Loading classifier model done!")

    while True:
        print("Enter sentence:")
        sentence = str(input())
        pred = predictor.predict(sentence)
        print(pred[0])
