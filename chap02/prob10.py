import pandas as pd

"""
with open("./data/popular-names.txt") as f:
    lines = f.readlines()
    print(len(lines))
"""

popular_name_df = pd.read_csv("./data/popular-names.txt", header=None, sep='\t')
print(popular_name_df.shape[0])