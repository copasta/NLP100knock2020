import pandas as pd
import re

df = pd.read_json("./data/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_text = uk_text.split("\n")
uk_section = [line for line in uk_text if "==" in line]
uk_dict = {}
for section_line in uk_section:
    section_name = re.sub('=| ', '', section_line)
    uk_dict[section_name] = int(section_line.count('=')/2 -1)
print(uk_dict)