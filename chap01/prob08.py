def cipher(text):
    return_list = []
    for char in text:
        
        # 文字コードへ変換
        char_ord = ord(char)

        if char_ord >= ord("a") and char_ord <= ord("z"):
            return_list.append(chr(219 - char_ord))
        else:
            return_list.append(char)
    
    return "".join(return_list)

def main():
    text = "I am an NLPer."
    print('original  :', text)
    print('encryption:', cipher(text))
    print('decryption:', cipher(cipher(text)))

if __name__ == "__main__":
    main()