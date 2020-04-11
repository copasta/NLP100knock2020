from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'IPAexGothic'


def get_morpheme(mecab_list):
    list_morpheme = []
    for sentence in mecab_list:
        for line in sentence.split("\n"):
            if line == "":
                continue
            line = line.split("\t")
            surface = line[0]
            res = line[1].split(",")
            dict_morpheme = {
                "surface" : surface,
                "base" : res[6],
                "pos" : res[0],
                "pos1" : res[1]
            }
            list_morpheme.append(dict_morpheme)

    return list_morpheme

def main():
    with open("./data/neko.txt.mecab", mode="rt", encoding="utf-8") as f:
        list_neko = f.read().split("EOS\n")
    list_morpheme_neko = get_morpheme(list_neko)
    list_base_neko = [dict_morpheme["base"] for dict_morpheme in list_morpheme_neko]
    count_neko = defaultdict(int)
    for idx in range(len(list_base_neko)):
        if idx == 0:
            if list_base_neko[idx+1] == "猫":
                count_neko[list_base_neko[idx]] += 1
        elif idx == len(list_base_neko)-1:
            if list_base_neko[idx-1] == "猫":
                count_neko[list_base_neko[idx]] += 1
        else:
            if list_base_neko[idx+1] == "猫" or list_base_neko[idx-1] == "猫":
                count_neko[list_base_neko[idx]] += 1
    count_neko = sorted(count_neko.items(), key=lambda x: x[1], reverse=True)

    word = [pair[0] for pair in count_neko[:10]]
    freq = [pair[1] for pair in count_neko[:10]]

    plt.figure(figsize=(5, 5))
    plt.bar(word, freq)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
