import matplotlib.pyplot as plt
from preprocessing import readData, TASK_1, EXTRA_TRAIN_TASK_1, TASK_2, EXTRA_TRAIN_TASK_2, readDataTask2
import numpy as np

def task1():
    data = readData([TASK_1 / 'train.csv', TASK_1 / 'dev.csv', EXTRA_TRAIN_TASK_1])

    scores = [score for _, _, score in data]
    plt.hist(scores)
    plt.show()


def task2():
    data2 = readDataTask2([TASK_2 / 'train.csv', TASK_2 / 'dev.csv', EXTRA_TRAIN_TASK_2])
    labels = [label for _, _, _, _, label in data2]
    plt.hist(labels, bins=np.arange(4)-0.5)
    plt.xticks(np.arange(3), np.arange(3))
    plt.show()
    
if __name__ == '__main__':
    task2()
