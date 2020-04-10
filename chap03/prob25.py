import re

import pandas as pd

df = pd.read_json("./data/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_text = uk_text.split("\n")

basic_info = {}

for text in uk_text:
    if ' = ' in text:
        text = text[1:].split(' = ')
        basic_info[text[0]] = text[1]

for k, v in basic_info.items():
    print("{}:{}".format(k, v))