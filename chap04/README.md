# MeCabインストール(Mac)
簡単なMeCabと辞書のインストール手順(2019-04-03)

## MeCabのインストール
```
$ brew install mecab
$ brew install mecab-ipadic

# Python3からMeCabが扱えるライブラリ
$ pip install mecab-python3

# 動作確認
$ python -c "import MeCab"
```

## neologd辞書のインストール
```
$ brew install git curl xz
$ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
$ cd mecab-ipadic-neologd
$ ./bin/install-mecab-ipadic-neologd -n -a

# defaultの辞書をneologd辞書にする
$ cd /usr/local/etc
$ open mecabrc

# 4行目に以下の変更を行う
; dicdir =  /usr/local/lib/mecab/dic/ipadic 
dicdir =  /usr/local/lib/mecab/dic/mecab-ipadic-neologd 
```

