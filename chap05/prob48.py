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
    list_cunk_neko = get_chunk(list_neko)
    for batch_chunk in list_cunk_neko[:10]:
        for line_chunk in batch_chunk:
            list_path_text = []
            pos_dst = [line_morphs.pos for line_morphs in line_chunk.morphs]
            if "名詞" in pos_dst and line_chunk.dst != -1:
                list_path_text.append("".join([line_morphs.surface for line_morphs in line_chunk.morphs]))
                next_line_chunk = batch_chunk[line_chunk.dst]
                while True:
                    list_path_text.append(''.join([line_morphs.surface for line_morphs in next_line_chunk.morphs]))
                    if next_line_chunk.dst == -1:
                        break
                    next_line_chunk = batch_chunk[next_line_chunk.dst]
                print(
                    "->".join(
                        list_path_text
                    )
                )

if __name__ == "__main__":
    main()