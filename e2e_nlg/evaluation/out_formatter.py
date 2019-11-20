import pandas as pd
import os
import re
from utils.log import logger

class OutputFormatter:
    '''
    Take predictions and list of reference mrs in order, and output predictions in same order
    '''
    def __init__(self, ordered_mrs_file, preds_file, out_folder):
        self.mrs_file = ordered_mrs_file
        self.preds_file = preds_file
        self.out_folder = out_folder

    def write_predictions(self, outfile="pred_e2e_nlg.txt"):
        with open(self.mrs_file,"r") as f:
            mrs = []
            for line in f.readlines():
                mrs.append(", ".join(sorted([item.strip() for item in line.split(", ")])))

        preds_df = pd.read_csv(self.preds_file,sep="\t", encoding="utf-8")
        for i,row in preds_df.iterrows():
            preds_df.loc[i,"input"]=", ".join(sorted([item.strip() for item in row["input"].split(", ")]))

        preds = []
        for mr in mrs:
            pred = preds_df[preds_df["input"]==mr]["output"].iloc[0]
            pred = re.sub(r" *([\.,\-])",r"\1",pred)
            pred = re.sub(r"\- +", "-", pred)
            pred = " ".join([item[0].upper() + item[1:] + ". " for item in pred.split(". ")])
            pred = re.sub(r"\. *\. *",". ",pred)
            pred = re.sub(r"  +", " ", pred)
            pred = pred.strip()
            preds.append(pred)

        path = os.path.join(self.out_folder,outfile)
        with open(path,"w") as f:
            f.write("\n".join(preds))
            logger.info("Predictions written to {}".format(path))
        return path