# coding=utf-8
import pandas as pd
import re


class MR_Formatter:
    def __init__(self, input_value):
        self.input_value = input_value
        self.attributes = []

    def set_food(self):
        food = None
        if "chinese" in self.input_value.lower():
            food = "Chinese"
        elif "english" in self.input_value.lower():
            food = "English"
        elif "japanese" in self.input_value.lower():
            food = "Japanese"
        elif "fast food" in self.input_value.lower():
            food = "Fast food"
        elif "indian" in self.input_value.lower():
            food = "Indian"
        elif "french" in self.input_value.lower():
            food = "French"
        elif "italian" in self.input_value.lower():
            food = "Italian"

        if food is not None:
            self.attributes.append("food[{}]".format(food))


    def set_customer_rating(self):
        pattern = r"^.*customer rating ([^,\.]+)[,\.]?.*$"
        if re.match(pattern, self.input_value):
            customer_rating = re.sub(pattern, r"\1", self.input_value)
            customer_rating = MR_Formatter.clean_value(customer_rating)
            self.attributes.append("customer rating[{}]".format(customer_rating))

    def set_near(self):
        pattern = r"^.*near ([^,\.]+)[,\.]?.*$"
        if re.match(pattern, self.input_value):
            self.attributes.append("near[yes]")

    def set_area(self):
        pattern = r"^.*area ([^,\.]+)[,\.]?.*$"
        if re.match(pattern, self.input_value):
            area = re.sub(pattern, r"\1", self.input_value)
            self.attributes.append("area[{}]".format(area.strip()))

    def set_family_friendly(self):
        family_friendly = None
        positive_pattern = r"^.*family friendly yes.*$"
        negative_pattern = r"^.*family friendly no.*$"
        if re.match(positive_pattern, self.input_value):
            family_friendly = "yes"
        elif re.match(negative_pattern, self.input_value):
            family_friendly = "no"
        if family_friendly is not None:
            self.attributes.append("familyFriendly[{}]".format(family_friendly))

    def set_price_range(self):
        pattern = r"^.*price range ([^,\.]+)[,\.]?.*$"
        if re.match(pattern, self.input_value):
            price_range = re.sub(pattern, r"\1", self.input_value)
            price_range = MR_Formatter.clean_value(price_range)
            self.attributes.append("priceRange[{}]".format(price_range))

    def set_name(self):
        self.attributes.append("name[XXX]")

    def set_venue_type(self):
        eat_type = None
        if " pub " in self.input_value:
            eat_type = "pub"
        elif " coffee shop " in self.input_value:
            eat_type = "coffee shop"
        elif " restaurant " in self.input_value:
            eat_type = "restaurant"

        if eat_type is not None:
            self.attributes.append("eatType[{}]".format(eat_type))

    @staticmethod
    def clean_value(v):
        v = re.sub(r"(.) *([0-9]+) *\- * ([0-9]+)", r"\1\2-\3", v)
        v = re.sub(r"(£) *([0-9]+)", r"\1\2", v)
        v = v.strip()
        return v

    def set_attributes(self):
        self.set_venue_type()
        self.set_food()
        self.set_price_range()
        self.set_customer_rating()
        self.set_name()
        self.set_area()
        self.set_family_friendly()
        self.set_near()

class MRs_Formatter:
    '''
        Convert input column to mr original representation
    '''

    def __init__(self,in_file,out_file):
        self.input_df = pd.read_csv(in_file,sep="\t",encoding="utf-8")
        self.input_df = self.input_df.astype(str)
        self.output_df = None
        self.output_path = out_file
        self.input_value = None


    def process_mrs(self):
        rows = []
        for i, row in self.input_df.iterrows():
            mr_formatter = MR_Formatter(row["input"])
            mr_formatter.set_attributes()
            new_row = {"input": ", ".join(sorted(mr_formatter.attributes)),"output": row["output"], "target": row["target"]}
            rows.append(new_row)

        self.output_df = pd.DataFrame(rows)

        self.output_df.to_csv(self.output_path, index=False, sep="\t", encoding="utf-8")