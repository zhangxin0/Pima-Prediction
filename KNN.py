from sklearn.preprocessing import normalize
import numpy as np
import read_data
import math
import csv
import time

train_data, validation_data, test_data = read_data.read_data()
train_data_sample = np.array(train_data[:][:-1])
y_train_sample = [train_data[i][-1] for i in range(len(train_data_sample))]

validation_data_sample = np.array(validation_data[:][:-1])
y_validation = [validation_data[i][-1] for i in range(len(validation_data))]

test_data_sample = np.array(test_data[:][:-1])
y_test = [test_data[i][-1] for i in range(len(test_data))]

# normalizaiton:
train_data_sample = normalize(train_data_sample, axis=0, norm='max')
validation_data_sample = normalize(validation_data_sample, axis=0, norm='max')
test_data_sample = normalize(test_data_sample, axis=0, norm='max')


def distance(data1, data2):
    sum = 0.
    for i in range(len(data1)):
        # sum += math.pow(data1[i] - data2[i], 2)
        sum += abs(data1[i] - data2[i])
        # return math.sqrt(sum)
    return sum


# KNN:
k = 7
right_num = 0
validated = {}
num = len(test_data_sample)
monitor = 0
res = []
start = time.perf_counter()
for i in range(num):
    # monitor:
    print(monitor)
    monitor += 1
    dis = []  # store (digit label, distance)
    for j in range(len(train_data_sample)):
        dis.append((y_train_sample[j], distance(test_data_sample[i], train_data_sample[j])))
    dis = sorted(dis, key=lambda item: item[1])
    # get the min 7 distance figs:
    neighbors = dis[:k]
    # dict (y,(nums,Min_dis))
    dict = {}
    vote = []
    for value, dis in neighbors:
        if value not in dict:
            dict[value] = [1, dis]
        else:
            dict[value][0] += 1
            if dis < dict[value][1]:
                dict[value][1] = dis
    # 选择投票数最多：vote[y,[nums,Min_dis]] dict_items = tuple(key,[value])
    vote = sorted(dict.items(), key=lambda item: item[1][0])
    vt_num_max = vote[-1][1][0]
    # if tie: choose the nearst one 先按照数量排序，再按照距离最近排序
    min_dis = vote[-1][1][1]
    best_predict = vote[-1][0]
    for value in dict:
        if dict[value][0] == vt_num_max and dict[value][1] < min_dis:
            min_dis = dict[value][1]
            best_predict = value
    res.append(best_predict)

num_right = 0
confusion_matrix = [[0] * 2 for i in range(2)]
for i in range(len(res)):
    value = int(y_test[i])
    if res[i] == value:
        confusion_matrix[value][value] += 1
        num_right += 1
    else:
        confusion_matrix[value][int(res[i])] += 1
accuracy = num_right / len(res)
end = time.perf_counter()
print('Run time:', end - start)
with open('confusion_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # data automatically write back with ',' split
    writer.writerow(['Accuracy:', accuracy])
    writer.writerow('Confusion Matrix:')
    writer.writerows(confusion_matrix)
