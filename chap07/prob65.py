

flag = False

with open('./data/prob64.txt') as f:
    for line in f.readlines():
        if line[0] == ":":
            if flag:
                print(acc / cnt)
            line = line.split()
            print(line[1])
            cnt = 0
            acc = 0
            flag = True
        else:
            line = line.split()
            cnt += 1
            if line[3] == line[4]:
                acc += 1
    print(acc / cnt)