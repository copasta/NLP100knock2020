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
        batch_temp = []
        noun_chunk = [line_chunk for line_chunk in batch_chunk if "名詞" in [line_morphs.pos for line_morphs in line_chunk.morphs]]
        for idx in range(len(noun_chunk) - 1):
            first_list_txt = ["".join([noun_morphs.surface if noun_morphs.pos != "名詞" else "X" for noun_morphs in noun_chunk[idx].morphs])]
            for next_noun_chunk in noun_chunk[idx+1:]:
                
                temp_next_line_text = []
                for noun_morphs in next_noun_chunk.morphs:
                    if noun_morphs.pos != "記号":
                        if noun_morphs.pos == "名詞":
                            temp_next_line_text.append("Y")
                        else:
                            temp_next_line_text.append(noun_morphs.surface)

                next_list_txt = ["".join(temp_next_line_text)]
                next_dst = noun_chunk[idx].dst
                next_chunk_idx = batch_chunk.index(next_noun_chunk)
                next_temp = []

                while True:
                    next_temp.append(batch_chunk[next_dst])
                    next_dst = batch_chunk[next_dst].dst
                    if next_dst == -1 or next_dst == next_chunk_idx:
                        break
                
                if len(next_temp) < 1 or next_temp[-1].dst != -1:
                    next_list_path = ["".join([next_line_morphs.surface for next_line_morphs in next_line_chunk.morphs if next_line_morphs.pos != '記号']) for next_line_chunk in next_temp]
                    batch_temp.append(" -> ".join(first_list_txt + next_list_path + ["Y"]).replace("XX", "X").replace("YY", "Y"))
                else:
                    next_dst = next_noun_chunk.dst
                    next_list_path = []
                    while True:
                        next_list_path.append("".join([line_morphs.surface for line_morphs in batch_chunk[next_dst].morphs if line_morphs.pos != "記号"]))
                        next_dst = batch_chunk[next_dst].dst
                        if batch_chunk[next_dst] in next_temp:
                            break
                    end_list_txt = ["".join([line_morphs.surface for line_morphs in batch_chunk[next_dst].morphs if line_morphs.pos != "記号"])]
                    if next_list_path[-1] != end_list_txt[0]:
                        batch_temp.append(" | ".join(first_list_txt + [" -> ".join(next_list_txt + next_list_path)] + end_list_txt).replace("XX", "X").replace("YY", "Y"))
                    else:
                        batch_temp.append(" | ".join(first_list_txt + [" -> ".join(next_list_txt + next_list_path[:-1])] + end_list_txt).replace("XX", "X").replace("YY", "Y"))

        for batch_path in batch_temp:
            print(batch_path)




if __name__ == "__main__":
    main()