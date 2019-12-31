# -*- coding: utf-8 -*-

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
            mr = re.sub(r"near\[[^\]]+\]", "near[yes]", mr)
            if mr not in self.mrs_dict.keys():
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
        value = re.sub(r"xx[a-z][a-z]+", " ",value)
        value = value.replace("chinese","Chinese")
        value = value.replace("english","English")
        value = value.replace("french","French")
        value = value.replace("japanese","Japanese")
        value = value.replace("italian","Italian")
        value = value.replace("indian","Indian")
        value = re.sub(r"  +", " ", value)
        value = re.sub(r"(.) *([0-9]+) *\- * ([0-9]+)", r"\1\2-\3", value)
        value = re.sub(r"(£) *([0-9]+)", r"\1\2", value)
        value = value.strip()
        return value

    @staticmethod
    def relexicalize_item(row, attr_name, src):
        if len(src) > 1:
            logger.info("More than one item: {}".format(len(src)))
        src = src[0]
        row["input"] = src
        row[attr_name] = ReLexicalizer.clean_str(row[attr_name])
        name = ReLexicalizer.get_attribute_value(src, "name")
        near = ReLexicalizer.get_attribute_value(src, "near")
        if name != "" and re.search("the xxx",row[attr_name]) is not None:
            row[attr_name] = re.sub("the xxx",name,row[attr_name],flags=re.I)
        elif name != "":
            row[attr_name] = re.sub("xxx",name,row[attr_name],flags=re.I)
        if near != "" and re.search("the yyy",row[attr_name]) is not None:
            row[attr_name] = re.sub("the yyy",near,row[attr_name],flags=re.I)
        elif near != "":
            row[attr_name] = re.sub("yyy",near,row[attr_name],flags=re.I)
        return row

    def relexicalize(self):
        rows = []
        for i, row in self.df.iterrows():
            if row["input"] not in self.mrs_dict.keys():
                print("Not in mrs dictionary:",row["input"])
            else:
                src = self.mrs_dict[row["input"]]
                row = ReLexicalizer.relexicalize_item(row, "output", src)
                row = ReLexicalizer.relexicalize_item(row, "target", src)
                src.pop(0) # we remove the lexicalized mr
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.outpath, index=False, sep="\t")
        logger.info("Wrote relexicalized data to {}".format(self.outpath))