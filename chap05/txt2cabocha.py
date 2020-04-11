import CaboCha

c = CaboCha.Parser()
with open('./data/neko.txt') as txt_file, open('./data/neko.txt.cabocha', 'w') as cabocha_file:
    for line in txt_file:
        tree = c.parse(line.lstrip())
        cabocha_file.write(tree.toString(CaboCha.CABOCHA_FORMAT_LATTICE))