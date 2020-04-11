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
    extract_neko = []
    for dict_morpheme in list_morpheme_neko:
        if dict_morpheme["pos"] == "動詞":
            extract_neko.append(dict_morpheme["surface"])
    
    print(extract_neko[:15])

if __name__ == "__main__":
    main()
