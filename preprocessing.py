from pathlib import Path
import copy

import pandas as pd
import numpy as np


# ----------------- Paths for data files -------------------------------------------------
TASK_1 = Path('semeval/semeval-2020-task-7-data-full/task-1/')
TASK_2 = Path('semeval/semeval-2020-task-7-data-full/task-2/')

EXTRA_TRAIN_TASK_1 = Path('semeval/semeval-2020-task-7-extra-training-data/task-1/train_funlines.csv')
EXTRA_TRAIN_TASK_2 = Path('semeval/semeval-2020-task-7-extra-training-data/task-2/train_funlines.csv')
# ----------------------------------------------------------------------------------------


def sentToTokens(sentence):
    '''
    Takes a sentence with a designated word to replace, and outputs a list of tokens
    and the indices to replace
    '''
    begin, end = sentence.strip().split('<')
    middle, end = end.split('/>')

    begin = begin.strip().split(' ') if len(begin) > 0 else []
    middle = middle.split(' ')
    end = end.strip().split(' ') if len(end) > 0 else []

    return begin + middle + end, len(begin), len(begin) + len(middle)



def readData(files):
    '''
    Reads data from given files and outputs the data as a list of tuples
    ((sentenceTokens, replaceStartIndex, replaceEndIndex), replacement word, score)

    Create new sentence by doing sentenceTokens[replaceStartIndex:replaceEndIndex] = [replacement]
    '''
    trainDF = pd.read_csv(files[0])
    for fileName in files[1:]:
        trainDF = trainDF.append(pd.read_csv(fileName), ignore_index=True)

    return zip((sentToTokens(sent) for sent in trainDF.original), trainDF.edit, trainDF.meanGrade)

'''
This returns the following training data:
Treat each data point (headline and replacement) as two separate data points: the original headline with a humor score
of 0, and the altered headline with a humor score associated with that data point.'''
def model1preprocessing(files):
    raw_data = readData(files)
    training_data = []
    for data_point in raw_data:
        (original_sentence, replStart, replEnd), repl, score = data_point
        new_sentence = copy.deepcopy(original_sentence)
        new_sentence[replStart:replEnd] = [repl]

        training_data.append((original_sentence, 0))
        training_data.append((new_sentence, score))
    return training_data

'''
This returns the following training data:
One data point is the original headline and the full altered headline concatenated together.
'''
def model2preprocessing(files):
    raw_data = readData(files)
    training_data = []
    for data_point in raw_data:
        (original_sentence, replStart, replEnd), repl, score = data_point
        new_sentence = copy.deepcopy(original_sentence)
        new_sentence[replStart:replEnd] = [repl]

        two_sentences = original_sentence + new_sentence
        training_data.append((two_sentences, score))
    return training_data

'''
This returns the following training data:
The model runs the original headline through one LSTM, the altered headline through a separate LSTM, then performs some
function on the two outputs to predict a score.
'''
def model3preprocessing(files):
    raw_data = readData(files)
    #original_training_data = []
    #new_training_data = []
    training_data = []
    for data_point in raw_data:
        (original_sentence, replStart, replEnd), repl, score = data_point
        new_sentence = copy.deepcopy(original_sentence)
        new_sentence[replStart:replEnd] = [repl]

        training_data.append((original_sentence, new_sentence, score))
    return training_data


if __name__ == '__main__':
    training_data = list(readData([TASK_1 / 'train.csv', EXTRA_TRAIN_TASK_1]))
    print(len(training_data))

    (sent, replStart, replEnd), repl, score = training_data[0]
    print(training_data[0])
    print(f'original: {sent}')
    altered = sent
    altered[replStart:replEnd] = [repl]
    print(f'altered: {altered}')

    print()
    print('Sherry Test')
    training_data = model2preprocessing([TASK_1 / 'train.csv', EXTRA_TRAIN_TASK_1])
    print(training_data[0])
