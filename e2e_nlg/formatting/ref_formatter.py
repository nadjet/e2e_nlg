import pandas as pd
import os
from utils.log import logger

class ReferenceFormatter:
    '''
    Printing meaning representations to a file, and grouped references to another, in same order
    '''
    def __init__(self,ref_file,out_folder):
        self.ref_file = ref_file
        self.out_folder = out_folder
        self.ref_groups = []
        self.mrs = []

    def set_mr_groups_and_mrs(self):
        mrs_df = pd.read_csv(self.ref_file,sep=",") # the reference
        groups = mrs_df.groupby("mr")
        self.mrs = [group[0] for group in groups]

        for group in groups:
            refs = []
            for i,row in group[1].iterrows():
                refs.append(row["ref"])
            self.ref_groups.append(refs)

    def save_ref_groups(self, filename = "ref_e2e_nlg.txt"):
        path = os.path.join(self.out_folder, filename)
        with open(path,"w") as f:
            first = True
            for refs in self.ref_groups:
                if not first:
                    f.write("\n")
                else:
                    first=False
                for ref in refs:
                    f.write(ref)
                    f.write("\n")
            logger.info("Saved grouped references to {}".format(path))
        return path

    def save_mrs(self, filename = "mr_e2e_nlg.txt"):
        path = os.path.join(self.out_folder, filename)
        with open(path,"w") as f:
            for mr in self.mrs:
                f.write(mr)
                f.write("\n")
            logger.info("Saved grouped references to {}".format(path))
        return path