class Morpph:
    def __init__(self, dict_morpheme):
        self.surface = dict_morpheme["surface"]
        self.base = dict_morpheme["base"]
        self.pos = dict_morpheme["pos"]
        self.pos1 = dict_morpheme["pos1"]

def get_morpheme(list_cabocha):
    list_morpheme = []
    for sentence in list_cabocha:
        for line in sentence.split("\n"):
            if line == "" or line[0] == "*":
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
            list_morpheme.append(Morpph(dict_morpheme))
    return list_morpheme

def main():
    with open("./data/neko.txt.cabocha", mode="rt", encoding="utf-8") as f:
        list_neko = f.read().split("EOS\n")
    list_morpheme_neko = get_morpheme(list_neko)
    for line_morpheme in list_morpheme_neko[:10]:
        print(vars(line_morpheme))

if __name__ == "__main__":
    main()