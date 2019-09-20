import csv
import random


def read_data():
    data = []
    flag = True
    with open('data/diabetes.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if flag:
                flag = False
                continue
            data.append(list(map(float, row)))
    random.shuffle(data)
    i = len(data)
    train_data = data[:int(0.7 * i * 0.95)]
    validation_data = data[int(0.7 * i * 0.95) + 1:int(0.7 * i)]
    test_data = data[int(0.7 * i) + 1:]
    print("data size:", i)
    print("train data size", 0.7 * i * 0.95)
    print("validation data size:", 0.7 * i * 0.05)
    print("test data size:", 0.3 * i)
    return train_data, validation_data, test_data
read_data()