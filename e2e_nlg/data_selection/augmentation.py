import re
import spacy
import itertools
import random



from e2e_nlg.classification.mr_predictor import MrPredictor
from e2e_nlg.data_selection.data_pair import DataPair
from utils.seq2seq_predict import PredictUtils
from utils.log import logger

class TextAugmentation:
    def __init__(self,main_path, dataset_path, df):
        self.mr_predictor = MrPredictor(main_path,dataset_path)
        self.df = df
        self.nlp = spacy.load("en_core_web_sm")
        self.new_pairs = []


    def predictions_valid(self,text,pairs):
        predictions = []
        eatType = False
        for pair in pairs:
            mrs = pair.get_generalized_mr()
            if "eatType" in mrs:
                eatType=True
            predictions.extend(mrs.split(","))
        predictions = list(set(predictions))
        trues = self.mr_predictor.predict(text)[0]
        trues = ",".join(set(str(trues).split(";")))
        true_pair = DataPair()
        true_pair.set_text(text)
        true_pair.set_mr(trues)
        true_pair = true_pair.get_generalized_mr()
        trues = true_pair.split(",")
        if "eatType" not in true_pair and eatType:
            trues.append("eatType[restaurant]")
        elif "eatType" in true_pair and not eatType:
            predictions.append("eatType[restaurant]")
        f_score = PredictUtils.calculate_fscore(trues, predictions)
        if f_score==1:
            return True
        else:
            return False


    @staticmethod
    def preprocess_pair(ref,mr):
        ref = re.sub(r"  +", " ",ref)  # more than one space in a row gives sometimes strange results in spacy
        name = re.sub(r"^.*name\[([^\]]+)\].*$", r"\1", mr.strip())
        near = re.sub(r"^.*near\[([^\]]+)\].*$", r"\1", mr.strip())
        if name==mr.strip():
            name=""
        if near== mr.strip():
            near=""
        # we replace pronouns at beginning of sentence with XXX and correct follow up auxiliary if relevant
        name_replacement = "Xxx"
        for k, v in [("They are",name_replacement+" is"),("They have",name_replacement+" has"),("They",name_replacement),("Their ",name_replacement+" "),("It ",name_replacement+" ")]:
            ref = ref.replace(k, v)
        return ref,name, near

    @staticmethod
    def postprocess_pair(ref,mr,name,near):
        mr = mr.split(",")
        if "Xxx" in ref:
            mr.append("name[" + name + "]")
        if "near[yes]" in mr:
            mr.remove("near[yes]")
            mr.append("near[" + near + "]")
        mr = ",".join(mr)
        ref = ref.replace("Xxx", name)
        ref = ref.replace("Yyy", near)
        return ref, mr


    def set_single_sentence_pairs(self):
        '''
        :return: extract new single sentence pairs
        '''
        num_texts=0
        num_sents=0
        for i, row in self.df.iterrows():
            if i%10==0:
                logger.info("\ni={},#texts={},#sents={}\n".format(i,num_texts,num_sents))
            mr = row["mr"]
            new_row = MrPredictor.process_row(row)
            new_row["ref"],name, near = TextAugmentation.preprocess_pair(new_row["ref"],mr)
            doc = self.nlp(new_row["ref"])
            pairs = []
            for sent in doc.sents:
                sent = str(sent)
                mrs = self.mr_predictor.predict(sent)[0]
                mrs = ",".join(sorted(str(mrs).split(";")))
                pair = DataPair.get_singleton(mrs,sent)
                pairs.append(pair)
            if len(pairs)==1:
                continue
            if not self.predictions_valid(new_row["ref"],pairs): # we ensure the sum of predictions for all sentences in the text is the same as the text predictions
                continue
            for j in range(len(pairs)):
                pair = pairs[j]
                ref, mrs = TextAugmentation.postprocess_pair(pair.get_text(),pair.get_mr(), name, near)
                pair.set_mr(mrs)
                if "name" not in mrs: # the MR has to include the name to be a complete independent text
                    continue
                pair.set_text(ref)
                self.new_pairs.append(pair)
                logger.info("{},{},{}".format(i,j,str(pair)))
            num_sents = num_sents + len(pairs)
            num_texts = num_texts + 1


    def set_mr_prediction(self):
        '''
        replace mr with its prediction
        '''
        counter = 0
        for i, row in self.df.iterrows():
            if i%10==0:
                logger.info("i={},#diff={}".format(i, counter))
            mr = row["mr"]
            old_pair = DataPair.get_singleton(mr,row["ref"])
            new_row = MrPredictor.process_row(row)
            new_row["ref"],name, near = TextAugmentation.preprocess_pair(new_row["ref"],mr)
            mrs = self.mr_predictor.predict(new_row["ref"])[0]
            mrs = ",".join(sorted(str(mrs).split(";")))
            ref, mrs = TextAugmentation.postprocess_pair(new_row["ref"],mrs, name, near)
            pair = DataPair.get_singleton(mrs,ref)
            if pair.get_ordered_mr()!=old_pair.get_ordered_mr() and len(DataPair.get_mr_dict(mrs.split(",")))>1:
                counter = counter+1
                self.new_pairs.append(pair)

    def set_permutations(self,num_permutations=3):
        '''
        for each pair, give N pairs with permuted mrs
        '''
        counter=0
        for i,row in self.df.iterrows():
            if i % 10 == 0:
                logger.info("i={},#diff={}".format(i, counter))
            mr_list = row["mr"].split(",")
            permutations = list(itertools.permutations(mr_list))
            random.shuffle(permutations)
            choices = permutations[0:num_permutations]
            print(choices)
            for choice in choices:
                counter=counter+1
                pair = DataPair.get_singleton(",".join(choice),row["ref"])
                self.new_pairs.append(pair)


import sys
import os
import pandas as pd
from csv import writer
if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    out_file = sys.argv[3]
    csv_file = os.path.join(dataset_path,"trainset.csv")

    df = pd.read_csv(csv_file,sep=",")
    aug = TextAugmentation(model_path, dataset_path, df)
    aug.set_permutations(num_permutations=3)
    with open(out_file,"w") as f:
        w = writer(f,delimiter=",")
        w.writerow(["mr","ref"])
        for pair in aug.new_pairs:
            w.writerow([pair.get_mr(),pair.get_text()])
