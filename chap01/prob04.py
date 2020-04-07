def main():
    text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    text = text.replace(",", "").replace(".", "")
    text = text.strip()
    text = text.split()
    word2pos = {}
    for idx, word in enumerate(text):
        if idx in [0, 4, 5, 6, 7, 8, 14, 15, 18]:
            word2pos[word[:1]] = idx + 1
        else:
            word2pos[word[:2]] = idx + 1
    
    print(word2pos)

if __name__ == "__main__":
    main()