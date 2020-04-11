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
    count_neko = defaultdict(int)
    for dict_morpheme in list_morpheme_neko:
        count_neko[dict_morpheme["base"]] += 1
    count_neko = sorted(count_neko.items(), key=lambda x: x[1], reverse=True)

    word = [pair[0] for pair in count_neko[:10]]
    freq = [pair[1] for pair in count_neko[:10]]

    plt.figure(figsize=(5, 5))
    plt.bar(word, freq)
    plt.show()

if __name__ == "__main__":
    main()
