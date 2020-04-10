import pandas as pd

df = pd.read_json("./data/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_text = uk_text.split("\n")
uk_category = [line for line in uk_text if "Category" in line]
print(uk_category)