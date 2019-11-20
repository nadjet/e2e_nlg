import pandas as pd
import os
from utils.log import logger

class OutFormatter:
    '''
    Convert output to VizSeq format
    '''
    def __init__(self,predictions_file,reference_file,output_path):
        self.predictions_df = pd.read_csv(predictions_file,sep="\t")
        self.mrs_df = pd.read_csv(reference_file,sep=",")
        self.output_path = output_path

    def transform(self,pred_out_file="pred_e2e_nlg.txt",src_out_file="src_e2e_nlg.txt"):
        groups = self.mrs_df.groupby("mr")
        groups = sorted(groups, key=lambda x: len(x[1]), reverse=True)

        mrs = [group[0] for group in groups]

        refs = []
        for mr in mrs:
            res = self.predictions_df.loc[self.predictions_df["mr"] == mr, "output"].iloc[0]
            refs.append(res)

        path = os.path.join(self.output_path,pred_out_file)
        with open(path, "w") as f:
            for ref in refs:
                f.write(ref)
                f.write("\n")
        logger.info("Outputs for VizSeq written to: {}".format(path))

        path = os.path.join(self.output_path,src_out_file)
        with open(path, "w") as f:
            for mr in mrs:
                f.write(mr)
                f.write("\n")
        logger.info("Sources for VizSeq written to: {}".format(path))