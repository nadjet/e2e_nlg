import pandas as pd
import random


class DfUtils:
    def __init__(self,csv_file,sep=","):
        random.seed(42)
        self.df = pd.read_csvc(csv_file)

    def sample_group(self,column_name="mr",ratio=0.1):
        '''

        :param column_name:
        :param ratio: value >0 and <=1: ratio of group items to pick
        :return: 10% of column name groups, randomly selected
        '''

        values = list(set(self.df[column_name]))
        random.shuffle(values)
        number_items =int((len(values)*ratio))
        values = values[:number_items]
        rows = []
        for i,row in self.df.iterrows():
            if row[column_name] in values:
                rows.append(row)
        return pd.DataFrame(rows)
