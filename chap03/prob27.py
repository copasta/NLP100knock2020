import re

import pandas as pd

df = pd.read_json("./data/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_text = uk_text.split("\n")

basic_info = {}

for text in uk_text:
    if ' = ' in text:
        text = text[1:].split(' = ')
        text_removed = re.sub('\'+?','',text[1])
        if 'ファイル' not in text_removed:
            text_removed = re.sub(r'\[+?', '', text_removed)
            text_removed = re.sub(r'\]+?', '', text_removed)
        basic_info[text[0]] = text_removed

for k, v in basic_info.items():
    print("{}:{}".format(k, v))