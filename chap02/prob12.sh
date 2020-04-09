cut -f 1 ./data/popular-names.txt > ./data/col1_test.txt
diff ./data/col1.txt ./data/col1_test.txt
cut -f 2 ./data/popular-names.txt > ./data/col2_test.txt
diff ./data/col2.txt ./data/col2_test.txt