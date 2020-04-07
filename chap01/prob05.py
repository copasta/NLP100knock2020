def n_gram(text, n, mode="word", do_lower=False):

    if mode not in ["char", "word"]:
        raise(ValueError("this mode is not define."))
    
    if do_lower:
        text = text.lower()
    
    text = text.replace(",", " ,").replace(".", " .")
    if mode == "char":
        text = "".join(text.split())
        text = [text[idx] for idx in range(len(text))]
    else:
        text = text.split()
    
    return_list = []

    for idx in range(len(text)):
        if idx + n - 1 < len(text):
            return_list.append("".join(text[idx:idx+n]))
    
    return return_list

def main():
    text = "I am an NLPer"
    text_char_2 = n_gram(text, 2, mode="char")
    text_word_2 = n_gram(text, 2, mode="word")
    print("text :", text)
    print("char bi-gram :", text_char_2)
    print("word bi-gram :", text_word_2)

if __name__ == "__main__":
    main()
