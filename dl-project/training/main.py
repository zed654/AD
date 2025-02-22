import data.data_extract.main as data_extract_main


def main():
    data_extract_main.main()
    print(data_extract_main.BagParser.x_min)


if __name__ == "__main__":
    main()

