import re

class DataPair:
    def __init__(self,mr_sep=","):
        self.mr = None
        self.text = None
        self.mr_sep=mr_sep
        self.mr_order = ["name", "eatType", "food", "priceRange", "customer rating", "area", "familyFriendly", "near"]

    def __str__(self):
        return "{} {}".format(self.mr,self.text)

    def set_mr(self, mr: str):
        self.mr = mr

    def set_text(self, text: str):
        self.text = text

    @staticmethod
    def get_mr_dict(mr_list):
        mr_dict = {}
        for mr in mr_list:
            name = re.sub(r"^([^\[]+)\[([^\]]+)\]$", r"\1", mr.strip())
            value = re.sub(r"^([^\[]+)\[([^\]]+)\]$", r"\2", mr.strip())
            mr_dict[name]=value
        return mr_dict

    def get_ordered_mr(self):
        mrs = self.mr.split(self.mr_sep)
        mrs = DataPair.get_mr_dict(mrs)
        ordered_mr = []
        for name in self.mr_order:
            if name in mrs:
                ordered_mr.append(name+"["+mrs[name]+"]")
        return (self.mr_sep).join(ordered_mr)

    def get_mr(self):
        return self.mr

    def get_text(self):
        return self.text


    @staticmethod
    def get_singleton(mr, text):
        data_model = DataPair()
        data_model.set_mr(mr)
        data_model.set_text(text)
        return data_model

    def get_generalized_mr(self):
        new_mrs = []
        mrs = self.mr.split(self.mr_sep)
        for item in mrs:
            item = item.replace("less than £20", "1")
            item = item.replace("more than £30", "3")
            item = item.replace("£20-25", "2")
            item = item.replace("1 out of 5", "1")
            item = item.replace("3 out of 5", "2")
            item = item.replace("5 out of 5", "3")
            item = item.replace("low", "1")
            item = item.replace("average", "2")
            item = item.replace("moderate", "2")
            item = item.replace("cheap", "1")
            item = item.replace("high", "3")
            new_mrs.append(item)
        return (self.mr_sep).join(new_mrs)



import sys
import pandas as pd
from utils.log import logger
if __name__ == "__main__":
    csv_file = sys.argv[1]
    out_file = sys.argv[2]
    df = pd.read_csv(csv_file,sep=",")
    rows = []
    for i,row in df.iterrows():
        if i%100==0:
            logger.info(i)
        pair = DataPair()
        pair.set_mr(row["mr"])
        mr = pair.get_ordered_mr()
        ref = row["ref"]
        rows.append({"mr":mr,"ref":ref})
    df = pd.DataFrame(rows)
    df.to_csv(out_file,sep=",",index=False)