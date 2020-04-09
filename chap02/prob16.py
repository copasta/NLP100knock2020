import argparse
import pandas as pd
import math

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="length")
args = parser.parse_args()
N = int(args.n)

popular_name_df = pd.read_csv("./data/popular-names.txt", header=None, sep='\t')
split_num = math.ceil(popular_name_df.shape[0] / N)

for iter_num in range(N):
    popular_name_df.loc[iter_num * split_num : (iter_num + 1) * split_num].to_csv(f"./data/prob16_{iter_num}.txt", sep='\t', index=False, header=None)