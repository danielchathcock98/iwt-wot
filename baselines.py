import pandas as pd
import numpy as np
import math
from scipy import stats

from preprocessing import TASK_1, TASK_2, EXTRA_TRAIN_TASK_1, EXTRA_TRAIN_TASK_2, readData, readDataTask2


def mode_baseline(files):
    raw_data = readData(files)
    all_scores = []
    for data_point in raw_data:
        (original_sentence, replStart, replEnd), repl, score = data_point
        all_scores.append(score)
    return stats.mode(all_scores)[0][0]

def mean_baseline(files):
    raw_data = readData(files)
    all_scores = []
    for data_point in raw_data:
        (_, _, _), _, score = data_point
        all_scores.append(score)
    return sum(all_scores) / len(all_scores)

def task2_baseline(files):
    raw_data = readDataTask2(files)
    all_labels = []
    for data_point in raw_data:
        (sent1, replStart1, replEnd1), repl1, (sent2, replStart2, replEnd2), repl2, label = data_point
        all_labels.append(label)
    return stats.mode(all_labels)[0][0]
    

def evaluate_baseline(baseline_score, files):
    print(baseline_score)
    raw_data = readData(files)

    square_error = 0
    val_examples = 0
    for data_point in raw_data:
        (original_sentence, replStart, replEnd), repl, score = data_point
        square_error += (baseline_score - score) ** 2
        val_examples += 1

    rmse = math.sqrt(square_error / val_examples)
    return rmse

def evaluate_baseline_task2(baseline_score, files):
    print(baseline_score)
    raw_data = readDataTask2(files)

    num_correct = 0
    val_examples = 0
    for data_point in raw_data:
        (sent1, replStart1, replEnd1), repl1, (sent2, replStart2, replEnd2), repl2, label = data_point
        if label == baseline_score:
            num_correct += 1
        val_examples += 1

    return num_correct/val_examples

if __name__ == '__main__':
    #print('mode baseline:')
    #print(evaluate_baseline(mode_baseline([TASK_1 / 'train.csv', EXTRA_TRAIN_TASK_1, TASK_1 / 'dev.csv']), [TASK_1 / 'test.csv']))

    #print('mean baseline:')
    #print(evaluate_baseline(mean_baseline([TASK_1 / 'train.csv', EXTRA_TRAIN_TASK_1, TASK_1 / 'dev.csv']), [TASK_1 / 'test.csv']))

    print('task2 baseline')
    print(evaluate_baseline_task2(task2_baseline([TASK_2 / 'train.csv', EXTRA_TRAIN_TASK_2, TASK_2 / 'dev.csv']), [TASK_2 / 'test.csv']))

