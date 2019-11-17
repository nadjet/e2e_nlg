import pandas as pd
import os
import re


class E2ENLGCleanedDataset:
    def __init__(self,path,filename,delimiter=","):
        self.df = pd.read_csv(os.path.join(path,filename),sep=delimiter)

    @staticmethod
    def make_mr(attr, attrs):
        if attr in attrs.keys():
            if attr == "name" and (attrs[attr].startswith("The ") or attrs[attr].startswith("the ")):
                return "The XXX"
            elif attr == "name":
                return "XXX"
            if attr == "eatType":
                return attrs[attr]
            elif attr == "food":
                return attrs[attr]
            elif attr == "customer rating":
                return "customer rating " + attrs[attr]
            elif attr == "priceRange":
                return "price range " + attrs[attr]
            elif attr == "area":
                return "area " + attrs[attr]
            elif attr == "familyFriendly":
                return "family friendly " + attrs[attr]
            elif attr == "near" and (attrs[attr].startswith("The") or attrs[attr].startswith("the")):
                return "near the YYY"
            elif attr == "near":
                return "near YYY"
        else:
            return None

    @staticmethod
    def make_mrs(attrs):
        sents = []
        names = ["name","eatType", "food", "customer rating", "priceRange", "familyFriendly", "area", "near"]
        for name in names:
            sent = E2ENLGCleanedDataset.make_mr(name, attrs)
            if sent is not None:
                sents.append(sent)
        return ", ".join(sents)

    def clean(self,is_ref=True):
        rows = []
        money_pattern = r"([^a-zA-Z0-9]?) *([0-9]+) *([^a-zA-Z0-9]?)"
        money_replace = r" \1 \2 \3"
        for i, row in self.df.iterrows():
            if is_ref:
                new_row = {"old_MR": row["mr"], "old_ref": row["ref"]}
                row["ref"] = re.sub(money_pattern,money_replace,row["ref"])
                new_row["ref"] = row["ref"]
                new_row["ref"] = re.sub(r"  +", " ", new_row["ref"]).strip()
            else:
                new_row = {"old_MR": row["mr"]}
            attributes = row['mr'].split(",")
            attribute_values = {}
            for attribute in attributes:
                attribute = attribute.strip()
                name = re.sub(r"^(.+)\[.+$", r"\1", attribute)
                value = re.sub(r"^.+\[(.+)\]$", r"\1", attribute)
                attribute_values[name] = value
                if is_ref and name == "name" and (value.startswith("the ") or value.startswith("The ")):
                    new_row["ref"] = new_row["ref"].replace(value, "The XXX")
                elif is_ref and name == "near" and (value.startswith("the ") or value.startswith("The ")):
                    new_row["ref"] = new_row["ref"].replace(value, "The YYY")
                if is_ref and name == "name":
                    new_row["ref"] = new_row["ref"].replace(value, "XXX")
                elif is_ref and name == "near":
                    new_row["ref"] = new_row["ref"].replace(value, "YYY")
            new_row["mr"] = E2ENLGCleanedDataset.make_mrs(attribute_values)
            new_row["mr"] = re.sub(money_pattern, money_replace, new_row["mr"])
            new_row["mr"] = re.sub(r"  +"," ",new_row["mr"]).strip()
            rows.append(new_row)

        self.df = pd.DataFrame(rows)