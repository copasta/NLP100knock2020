import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="length")
args = parser.parse_args()
N = int(args.n)

popular_name_df = pd.read_csv("./data/popular-names.txt", header=None, sep='\t')
print(popular_name_df.head(N))
