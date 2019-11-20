from lexicalrichness import LexicalRichness
import pandas as pd

with open("/Users/nadjet/Work/seq2seq/gitignore/pred_e2e_nlg.txt","r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = " ".join(lines)

lex = LexicalRichness(lines)


print(lex.words,lex.terms,lex.ttr)

df = pd.read_csv("/Users/nadjet/Work/seq2seq/gitignore/devset.csv")

groups = df.groupby("mr")

items = []
for group in groups:
    item = group[1].sample(1)
    items.append(item.iloc[0]["ref"])

lex = LexicalRichness(" ".join(items))

print(lex.words,lex.terms,lex.ttr)