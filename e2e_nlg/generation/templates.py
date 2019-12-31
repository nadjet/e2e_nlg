import pandas as pd
import re
from utils.log import logger

class TemplateGenerator:
    '''
        This class generate for each meaning representation a template based sentence of the form:
        [name] is a [familyFriendly] [eatType] which serves [food] food in the [price] price range. It has a [customerRating] customer rating. It is located in the [area] area, near [near].
    '''
    def __init__(self, csv_file, out_path):
        self.df = pd.read_csv(csv_file,sep=",")
        self.out_path = out_path

    @staticmethod
    def build_mrs_dict(text):
        attributes = text.split(",")
        mrs_dict = {}
        for name_value in attributes:
            name_value = name_value.strip()
            name = re.sub(r"^(.+)\[(.+)\]$",r"\1",name_value)
            value = re.sub(r"^(.+)\[(.+)\]$",r"\2",name_value)
            mrs_dict[name]=value
        return mrs_dict

    @staticmethod
    def generateFamilyFriendly(mrs_dict):
        if "familyFriendly" in mrs_dict and mrs_dict["familyFriendly"]=="yes":
            return ["a family friendly"]
        elif "familyFriendly" in mrs_dict:
            return ["a non family friendly"]
        else:
            return []

    @staticmethod
    def generateFood(mrs_dict):
        text = []
        if "food" in mrs_dict:
            text.extend(["which","serves",mrs_dict["food"]])
            if "food" not in mrs_dict["food"]:
                text.append("food")
        return text

    @staticmethod
    def generateEatType(mrs_dict,previous):
        text = []
        if previous == []:
            text.append("a")
        if "eatType" in mrs_dict:
            text.append(mrs_dict["eatType"])
        else:
            text.append("restaurant")
        return text


    @staticmethod
    def generateArea(mrs_dict):
        text = []
        if "area" in mrs_dict:
            text = ["It is located in the",mrs_dict["area"],"area"]
        return text

    @staticmethod
    def generateNear(mrs_dict, area):
        if "near" in mrs_dict and area != []:
            return [", near",mrs_dict["near"],"."]
        elif "near" in mrs_dict:
            return ["It is located near",mrs_dict["near"],"."]
        elif area != []:
            return ["."]
        return []

    @staticmethod
    def generateCustomerRating(mrs_dict):
        if not "customer rating" in mrs_dict:
            return []
        elif re.match(r"^[0-9].*",mrs_dict["customer rating"]):
            return ["It has a customer rating of",mrs_dict["customer rating"],"."]
        elif re.match(r"^[AEIOUaeiou].+",mrs_dict["customer rating"]):
            return ["It has an",mrs_dict["customer rating"],"customer rating."]
        else:
            return ["It has a",mrs_dict["customer rating"],"customer rating."]

    @staticmethod
    def generatePriceRange(mrs_dict):
        if not "priceRange" in mrs_dict :
            return []
        elif not re.match(r".*[0-9].*",mrs_dict["priceRange"]):
            return ["in the",mrs_dict["priceRange"],"price range"]
        else:
            return ["in the price range of",mrs_dict["priceRange"]]

    def generate(self):
        self.df["template"]=""
        for i,row in self.df.iterrows():
            if i%10==0:
                logger.info(i)
            mrs_dict = TemplateGenerator.build_mrs_dict(row["mr"])
            text = [mrs_dict["name"],"is"]
            family_friendly = TemplateGenerator.generateFamilyFriendly(mrs_dict)
            text.extend(family_friendly)
            eat_type = TemplateGenerator.generateEatType(mrs_dict,family_friendly)
            text.extend(eat_type)
            food = TemplateGenerator.generateFood(mrs_dict)
            text.extend(food)
            price_range = TemplateGenerator.generatePriceRange(mrs_dict)
            text.extend(price_range)
            text.append(".")
            customer_rating = TemplateGenerator.generateCustomerRating(mrs_dict)
            text.extend(customer_rating)
            area = TemplateGenerator.generateArea(mrs_dict)
            text.extend(area)
            near = TemplateGenerator.generateNear(mrs_dict, area)
            text.extend(near)
            text = " ".join(text)
            text = re.sub(r" \.",".",text)
            text = re.sub(r" ,", ",", text)
            text = re.sub(r"  +"," ",text)
            self.df.loc[i,"template"] = text

    def save_to_eval(self):
        ref_file = os.path.join(self.out_path,"ref_e2e_nlg.txt")
        mr_file = os.path.join(self.out_path,"mr_e2e_nlg.txt")
        pred_file = os.path.join(self.out_path,"pred_e2e_nlg.txt")
        with open(ref_file,"w") as ref_w, open(mr_file,"w") as mr_w, open(pred_file,"w") as pred_w:
            for mr, gr in self.df.groupby('mr'):
                mr_w.write(mr + "\n")
                first = True
                for i,row in gr.iterrows():
                    if first:
                        pred_w.write(row["template"] + "\n")
                        first = False
                    ref_w.write(row["ref"]+"\n")
                ref_w.write("\n")

import sys
import os
if __name__ == "__main__":
    csv_file = sys.argv[1]
    out_path = sys.argv[2]
    template_generator = TemplateGenerator(csv_file,out_path)
    template_generator.generate()
    template_generator.save_to_eval()

