from collections import Counter


if __name__ == "__main__":
    lines = ""
    file_path = '/Users/seonuk/PycharmProjects/pythonProject2/OBS_ASOS_DD_20240107222617.csv'
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    print(len(lines))

    len_list=[]
    for _ in range(len(lines)):
        l = len(lines[_].split(','))
        if l != 8:
            print(str(_) + ":" + str(l))
        len_list.append(len(lines[_].split(',')))

    Counter(len_list)