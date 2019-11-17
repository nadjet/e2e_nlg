# coding=utf-8
import pandas as pd
import os
import re

class MR_Formatter:
    '''
        Convert input column to mr original representation
    '''

    def __init__(self,in_file,out_file):
        self.input_df = pd.read_csv(in_file,sep="\t",encoding="utf-8")
        self.output_df = None
        self.output_path = out_file
        self.input_value = None

    def get_food(self):
        if "chinese" in self.input_value:
            return "Chinese"
        elif "english" in self.input_value:
            return "English"
        elif "japanese" in self.input_value:
            return "Japanese"
        elif "fast food" in self.input_value:
            return "Fast food"
        elif "indian" in self.input_value:
            return "Indian"
        elif "french" in self.input_value:
            return "French"
        elif "italian" in self.input_value:
            return "Italian"
        return ""


    def get_customer_rating(self):
        pattern = r"^.*customer rating is ([^\.]+)\..*$"
        if re.match(pattern, self.input_value):
            return re.sub(pattern, r"\1", self.input_value)
        return ""


    def get_near(self):
        pattern = r"^.*near ([^\.]+)\..*$"
        if re.match(pattern, self.input_value):
            return "YYY"
        return ""


    def get_area(self):
        pattern = r"^.*located in the ([^\.]+)\..*$"
        if re.match(pattern, self.input_value):
            return re.sub(pattern, r"\1", self.input_value)
        return ""


    def get_family_friendly(self):
        positive_pattern = r"^.*is family friendly.*$"
        negative_pattern = r"^.*is not family friendly.*$"
        if re.match(positive_pattern, self.input_value):
            return "yes"
        elif re.match(negative_pattern, self.input_value):
            return "no"
        return ""


    def get_price_range(self):
        pattern = r"^.*price range is ([^\.]+)\..*$"
        if re.match(pattern, self.input_value):
            return re.sub(pattern, r"\1", self.input_value)
        return ""


    def get_venue_type(self):
        if "a pub." in self.input_value:
            return "pub"
        elif "a coffee shop" in self.input_value:
            return "coffee shop"
        elif "a restaurant." in self.input_value:
            return "restaurant"
        return ""


    @staticmethod
    def clean_value(v):
        v = re.sub(r"(.) *([0-9]+) *\- * ([0-9]+)", r"\1\2-\3", v)
        v = re.sub(r"(£) *([0-9]+)", r"\1\2", v)
        v = v.strip()
        return v

    def process_mr(self):
        rows = []
        for i, row in self.input_df.iterrows():
            # attributes in order
            self.input_value = row["input"]
            attributes = ["name[XXX]"]

            vtype = MR_Formatter.clean_value(self.get_venue_type())
            if vtype != "":
                attributes.append("eatType[{}]".format(vtype))

            food = MR_Formatter.clean_value(self.get_food())
            if food != "":
                attributes.append("food[{}]".format(food))

            prange = MR_Formatter.clean_value(self.get_price_range())
            if prange != "":
                attributes.append("priceRange[{}]".format(prange))

            crating = MR_Formatter.clean_value(self.get_customer_rating())
            if crating != "":
                attributes.append("customer rating[{}]".format(crating))

            area = MR_Formatter.clean_value(self.get_area())
            if area != "":
                attributes.append("area[{}]".format(area))

            ffriendly = MR_Formatter.clean_value(self.get_family_friendly())
            if ffriendly != "":
                attributes.append("familyFriendly[{}]".format(ffriendly))

            near = MR_Formatter.clean_value(self.get_near())
            if near != "":
                attributes.append("near[{}]".format(near))

            print(row["input"], ", ".join(attributes))

            new_row = {"input": ", ".join(attributes), "output": row["output"], "target": row["target"]}

            rows.append(new_row)

        self.output_df = pd.DataFrame(rows)

        self.output_df.to_csv(os.path.join(self.output_path, "mr_e2e2_nlg.csv"), index=False, sep="\t", encoding="utf-8")