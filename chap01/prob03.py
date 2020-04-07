
def main():
    text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    text = text.replace(",", "").replace(".", "")
    text = text.strip()
    text_len = list(map(len, text.split()))
    print(text_len)

if __name__ == "__main__":
    main()
