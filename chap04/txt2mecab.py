import MeCab

with open("./data/neko.txt") as txt_file, open("./data/neko.txt.mecab", "w") as mecab_file:
    mecab = MeCab.Tagger()
    mecab_file.write(mecab.parse(txt_file.read()))