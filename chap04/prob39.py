from collections import defaultdict

import matplotlib.pyplot as plt

plt.style.use('ggplot')


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
    word_rank = []
    word_freq = []
    for idx, pair in enumerate(count_neko):
        word_rank.append(idx+1)
        word_freq.append(pair[1])
    
    plt.figure(figsize=(7, 5))
    plt.scatter(
        word_rank,
        word_freq
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's law")
    plt.show()

if __name__ == "__main__":
    main()
