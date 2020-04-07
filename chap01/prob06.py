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
    text_x = "paraparaparadise"
    text_y = "paragraph"
    X = set(n_gram(text_x, 2, mode="char"))
    Y = set(n_gram(text_y, 2, mode="char"))

    print("X", X)
    print("Y", Y)

    print()

    print("Union           :", X | Y)
    print("Intersection    :", X & Y)
    print("Difference(X-Y) :", X - Y)
    print("Difference(Y-X) :", Y - X)

    print()

    print("\'se\' in X :", "se" in X)
    print("\'se\' in Y :", "se" in Y)

if __name__ == "__main__":
    main()