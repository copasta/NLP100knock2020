import pandas as pd

popular_name_df = pd.read_csv("./data/popular-names.txt", header=None, sep='\t')
print(popular_name_df.sort_values(by=2, ascending=True))