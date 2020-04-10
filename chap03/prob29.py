import json
import re
from urllib.request import urlopen

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
        text_removed = re.sub(r'<ref(\s|>).+?(</ref>|$)', '', text_removed)
        text_removed = re.sub(r'<\s*?/*?\s*?br\s*?/*?\s*>', '', text_removed)
        basic_info[text[0]] = text_removed

flag_filename = basic_info["国旗画像"].replace(" ", "_")
flag_url = f'https://commons.wikimedia.org/w/api.php?action=query&titles=File:{flag_filename}&prop=imageinfo&iiprop=url&format=json'
flag_response = urlopen(flag_url)
flag_image_url = re.search(r'"url":"(.+?)"', flag_response.read().decode("utf-8")).group(1)
print(flag_image_url)
