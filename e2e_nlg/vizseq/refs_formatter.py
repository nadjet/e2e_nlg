import pandas as pd
import os
from utils.log import logger

class RefsFormatter:
    '''
    Write each reference to a file as required by Vizseq visualization tool
    '''
    def __init__(self,ref_file,out_path):
        self.ref_df = pd.read_csv(ref_file,sep=",")
        self.out_path = out_path

    def transform(self):
        groups = self.ref_df.groupby("mr")

        groups = sorted(groups, key=lambda x: len(x[1]), reverse=True)

        refs = []
        for group in groups:
            counter = 0
            for i, row in group[1].iterrows():
                if len(refs) == counter:
                    refs.append([])
                refs[counter].append(row["ref"])
                counter = counter + 1

        longest_ref = refs[0]
        for i in range(1, len(refs)):
            current_ref = refs[i]
            for j in range(0, len(longest_ref)):
                if j > len(current_ref) - 1:
                    current_ref.append(longest_ref[j])
            refs[i] = current_ref

        i = 0
        for ref in refs:
            path = os.path.join(self.out_path,"ref_"+str(i)+".txt")
            with open(path, "w") as f:
                for item in ref:
                    f.write(item)
                    f.write("\n")
                i = i + 1
        logger.info("References distributed amongst {} files in folder '{}'".format(len(refs),self.out_path))