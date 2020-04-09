import pandas as pd

popular_name_df = pd.read_csv("./data/popular-names.txt", header=None, sep='\t')
popular_name_df.to_csv("./data/prob11.txt", index=False, header=None, sep=" ")