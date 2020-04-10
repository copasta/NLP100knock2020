import re

import pandas as pd

df = pd.read_json("./data/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_media = re.findall(r'(?:File|ファイル):(.+?)\|', uk_text)
print(uk_media)
