import os
import re

import pandas as pd
from sklearn.utils import shuffle

MONEY_PATTERN = r"([^a-zA-Z0-9]?) *([0-9]+) *([^a-zA-Z0-9]?)"
MONEY_REPLACE = r" \1 \2 \3"


class E2ENLGCleanedDataset:

    def __init__(self, path, filename, delimiter=","):
        self.df = pd.read_csv(os.path.join(path, filename), sep=delimiter)
        self.df = shuffle(self.df, random_state=42)

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
        names = ["name", "eatType", "food", "customer rating", "priceRange", "familyFriendly", "area", "near"]
        for name in names:
            sent = E2ENLGCleanedDataset.make_mr(name, attrs)
            if sent is not None:
                sents.append(sent)
        return ", ".join(sents)

    @staticmethod
    def clean_ref(ref):
        ref = re.sub(MONEY_PATTERN, MONEY_REPLACE, ref)
        ref = re.sub(r"\-([a-zA-Z]+)", r" - \1", ref)  # we put a space between dashes


        ref = re.sub(r"([\.;,])([^ 0-9])", r"\1 \2", ref)  # we put a space after punctuation if there is not
        ref = re.sub(r"([;,] *)$", ".", ref)  # we replace end-of string comma or semi-column by period
        ref = ref.strip()
        ref = re.sub(r"([a-zA-Z])$", r"\1.", ref)  # we add a full stop if there is not any
        # ref = ContractionExpander.expand_contractions(ref)
        ref = re.sub(r"[\r\n\t]", " ", ref)
        ref = re.sub(r"  +", " ", ref)
        ref = re.sub(r"  +", " ", ref).strip()
        return ref


    @staticmethod
    def delexicalize_ref_value(ref, value_to_replace, value_replacing):
        if value_to_replace.startswith("the ") or value_to_replace.startswith("The "):
            ref = re.sub(value_to_replace, "The " + value_replacing, ref, flags=re.I)
        else:
            ref = re.sub(value_to_replace, value_replacing, ref, flags=re.I)
        return ref

    @staticmethod
    def delexicalize_ref(ref, attribute_values):
        if "name" in attribute_values:
            value = attribute_values["name"]
            ref = E2ENLGCleanedDataset.delexicalize_ref_value(ref, value, "XXX")
        if "near" in attribute_values:
            value = attribute_values["near"]
            ref = E2ENLGCleanedDataset.delexicalize_ref_value(ref, value, "YYY")
        return ref

    def clean(self, is_ref=True, is_mr=True):
        rows = []
        for i, row in self.df.iterrows():
            new_row = {}
            if is_ref:
                new_row["old_ref"] = row["ref"]
                new_row["ref"] = E2ENLGCleanedDataset.clean_ref(row["ref"])
            if is_mr:
                new_row["old_MR"] = row["mr"]
                attributes = row['mr'].split(",")
                attribute_values = {}
                for attribute in attributes:
                    attribute = attribute.strip()
                    name = re.sub(r"^(.+)\[.+$", r"\1", attribute)
                    value = re.sub(r"^.+\[(.+)\]$", r"\1", attribute)
                    value = value.strip()
                    name = name.strip()
                    attribute_values[name] = value
            if is_ref:
                new_row["ref"] = E2ENLGCleanedDataset.delexicalize_ref(new_row["ref"], attribute_values)
                # new_row["ref"] = deaccent(new_row["ref"])
            if is_mr:
                new_row["mr"] = E2ENLGCleanedDataset.make_mrs(attribute_values)
                new_row["mr"] = re.sub(MONEY_PATTERN, MONEY_REPLACE, new_row["mr"])
                new_row["mr"] = re.sub(r"  +", " ", new_row["mr"]).strip()
            rows.append(new_row)
        self.df = pd.DataFrame(rows)
