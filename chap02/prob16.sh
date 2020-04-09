#split -l 500 ./data/popular-names.txt -d --additional-suffix=.txt ./data/prob16_
n=`cat ./data/popular-names.txt | wc -l`
len=`expr $n / 5`
split -l $len ./data/popular-names.txt ./data/prob16_