import pandas as pd
import os
import sys
import re
import codecs
from utils.log import logger


class ReLexicalizer:
    '''
    Take as input the e2e output csv and the reference csv, and replace slots with lexical items, given mr as input
    '''

    def __init__(self, seq2seq_output, mrs_file, out_file):
        self.df = pd.read_csv(seq2seq_output, sep="\t", encoding="utf-8") # has input, output and target columns
        self.outpath = out_file
        self.mrs_path = mrs_file
        self.mrs_dict = {}

    def set_mrs_dict(self):
        df = pd.read_csv(self.mrs_path,sep=",")
        for i,row in df.iterrows():
            mr = ", ".join(sorted([item.strip() for item in row["mr"].split(",")]))
            original_mr = mr
            mr = re.sub(r"name\[[^\]]+\]", "name[XXX]", mr)
            mr = re.sub(r"near\[[^\]]+\]", "near[YYY]", mr)
            if original_mr not in self.mrs_dict.keys():
                self.mrs_dict[mr] = [original_mr]
            else:
                self.mrs_dict[mr].append(original_mr)

    @staticmethod
    def get_attribute_value(inp, attr_name):
        for item in inp.split(","):
            item = item.strip()
            if item.startswith(attr_name + "["):
                regex = attr_name + r"\[(.+)\]$"
                return re.sub(regex, r"\1", item)
        return ""

    @staticmethod
    def clean_str(value):
        value = value.replace("xxup", " ")
        value = value.replace("xxbos", " ")
        value = value.replace("xxmaj", " ")
        value = re.sub(r"  +", " ", value)
        value = re.sub(r"(.) *([0-9]+) *\- * ([0-9]+)", r"\1\2-\3", value)
        value = re.sub(r"(£) *([0-9]+)", r"\1\2", value)
        value = value.strip()
        return value

    def relexicalize_item(self, i, attr_name, src):
        if len(src) > 1:
            print("More than one item:",len(src))
        src = src[0]
        self.df.loc[i, "input"] = src
        self.df.loc[i, attr_name] = ReLexicalizer.clean_str(self.df.loc[i, attr_name])
        name = ReLexicalizer.get_attribute_value(src, "name")
        near = ReLexicalizer.get_attribute_value(src, "near")
        if name != "" and "the xxx" in self.df.loc[i, attr_name]:
            self.df.loc[i, attr_name] = self.df.loc[i, attr_name].replace("the xxx", name)
        elif name != "":
            self.df.loc[i, attr_name] = self.df.loc[i, attr_name].replace("xxx", name)
        if near != "" and "the yyy" in self.df.loc[i, attr_name]:
            self.df.loc[i, attr_name] = self.df.loc[i, attr_name].replace("the yyy", near)
        elif near != "":
            self.df.loc[i, attr_name] = self.df.loc[i, attr_name].replace("yyy", near)

    def relexicalize(self):
        self.set_mrs_dict()
        for i, row in self.df.iterrows():
            if row["input"] not in self.mrs_dict.keys():
                print("Not in mrs dictionary:",row["input"])
            else:
                src = self.mrs_dict[row["input"]]
                self.relexicalize_item(i, "output", src)
                self.relexicalize_item(i, "target", src)
        self.df.to_csv(self.outpath, index=False, sep="\t")
        logger.info("Wrote relexicalized data to {}".format(self.outpath))