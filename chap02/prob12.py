import pandas as pd

popular_name_df = pd.read_csv("./data/popular-names.txt", header=None, sep='\t')
popular_name_df[0].to_csv("./data/col1.txt", index=False, header=None)
popular_name_df[1].to_csv("./data/col2.txt", index=False, header=None)