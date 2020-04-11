# MeCabインストール(Mac)
簡単なMeCabと辞書のインストール手順(2019年版)

```
$ brew install mecab
$ brew install mecab-ipadic

$ brew install mecab mecab-ipadic git curl xz
$ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
$ cd mecab-ipadic-neologd
$ ./bin/install-mecab-ipadic-neologd -n -a

cd /usr/local/etc
open mecabrc

; dicdir =  /usr/local/lib/mecab/dic/ipadic 
dicdir =  /usr/local/lib/mecab/dic/mecab-ipadic-neologd 

$ pip install mecab-python3
```

