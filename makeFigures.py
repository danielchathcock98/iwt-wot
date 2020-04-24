import matplotlib.pyplot as plt
from preprocessing import readData, TASK_1, EXTRA_TRAIN_TASK_1

if __name__ == '__main__':
    data = readData([TASK_1 / 'train.csv', TASK_1 / 'dev.csv', EXTRA_TRAIN_TASK_1])

    scores = [score for _, _, score in data]
    plt.hist(scores)
    plt.show()
