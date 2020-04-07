def generate_template(x, y, z):
    return "{}時の{}は{}".format(x, y, z)

def main():
    x = 12
    y = "気温"
    z = 22.4

    sent = generate_template(x, y, z)

    print(sent)

if __name__ == "__main__":
    main()