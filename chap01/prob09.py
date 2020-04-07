import random

def generate_typoglycemia(text, seed=1234):
    text = text.split()

    random_instance = random.Random(seed)

    return_list = []
    for word in text:
        if len(word) > 4:
            word_first = word[0]
            word_last = word[-1]
            word_shuffle = list(word[1:-1])
            
            random_instance.shuffle(word_shuffle)

            return_list.append(word_first + "".join(word_shuffle) + word_last)
        else:
            return_list.append(word)
    
    return_text = " ".join(return_list)

    return return_text

def main():
    text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    text_typoglycemia = generate_typoglycemia(text)
    print("original text")
    print(text)
    print("typoglycemia text")
    print(text_typoglycemia)

if __name__ == "__main__":
    main()