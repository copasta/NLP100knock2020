import pandas as pd

col1_df = pd.read_csv("./data/col1.txt", header=None)
col2_df = pd.read_csv("./data/col2.txt", header=None)

df = pd.concat([col1_df, col2_df], axis=1)
df.to_csv('./data/prob13.txt', sep='\t', index=False, header=None)