# wc -l ./data/popular-names.txt

with open('./data/popular-names.txt') as f:
    lines = f.readlines()
    print(len(lines))