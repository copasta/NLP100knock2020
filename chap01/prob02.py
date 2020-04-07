def main():
    text1 = "パトカー"
    text2 = "タクシー"

    text_join = "".join([text1[idx] + text2[idx] for idx in range(len(text1))])

    print("text1 :", text1)
    print("text2 :", text2)
    print("text :", text_join)

if __name__ == "__main__":
    main()