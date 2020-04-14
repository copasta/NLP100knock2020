class Morpph:
    def __init__(self, dict_morpheme):
        self.surface = dict_morpheme["surface"]
        self.base = dict_morpheme["base"]
        self.pos = dict_morpheme["pos"]
        self.pos1 = dict_morpheme["pos1"]

class Chunk:
    def __init__(self, dst):
        self.morphs = [] 
        self.dst = dst # 係り先文節インデックス番号
        self.srcs = [] # 係り元文節インデックス番号

def get_chunk(list_cabocha):
    list_chunk = []
    temp = []
    for sentence in list_cabocha:
        for line in sentence.split("\n"):
            if line == "":
                continue
            elif line[0] == "*":
                dst = int(line.split()[2][:-1])
                chunk = Chunk(dst)
                temp.append(chunk)
            else:
                line = line.split("\t")
                surface = line[0]
                res = line[1].split(",")
                dict_morpheme = {
                    "surface" : surface,
                    "base" : res[6],
                    "pos" : res[0],
                    "pos1" : res[1]
                }
                chunk.morphs.append(Morpph(dict_morpheme))
        if len(temp):
            for idx, batch_chunk in enumerate(temp):
                temp[batch_chunk.dst].srcs.append(idx)
            list_chunk.append(temp)
            temp = []
    return list_chunk

def main():
    with open("./data/neko.txt.cabocha", mode="rt", encoding="utf-8") as f:
        list_neko = f.read().split("EOS\n")
    with open("./data/prob47.txt", mode="w") as f:
        list_cunk_neko = get_chunk(list_neko)
        for batch_chunk in list_cunk_neko:
            for batch_idx, line_chunk in enumerate(batch_chunk):
                word_dst = [line_morphs.surface for line_morphs in line_chunk.morphs]
                pos1_dst = [line_morphs.pos1 for line_morphs in line_chunk.morphs]
                if "を" in word_dst and "サ変接続" in pos1_dst:
                    for idx in range(len(word_dst) - 1):
                        if pos1_dst[idx] == "サ変接続" and word_dst[idx+1] == "を":
                            verb_dst = [line_morphs.base for line_morphs in batch_chunk[int(line_chunk.dst)].morphs if line_morphs.pos == "動詞"]
                            if verb_dst:
                                temp_pp = []
                                temp_phrase = []
                                for srcs in batch_chunk[int(line_chunk.dst)].srcs:
                                    if srcs == batch_idx or srcs == line_chunk.dst:
                                        continue
                                    temp_pp.append("".join([line_morphs.surface for line_morphs in batch_chunk[int(srcs)].morphs if line_morphs.pos == "助詞"]))
                                    temp_phrase.append("".join([line_morphs.surface for line_morphs in batch_chunk[int(srcs)].morphs if line_morphs.pos != "記号"]))
                                if temp_pp != [""] and len(temp_pp) > 0:
                                    txt = "{}\t{}\t{}".format(
                                            word_dst[idx] + word_dst[idx+1] + verb_dst[0],
                                            " ".join(temp_pp),
                                            " ".join(temp_phrase)
                                        )
                                    print(txt)
                                    f.write(txt + "\n")

if __name__ == "__main__":
    main()