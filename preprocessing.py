from pathlib import Path

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
        trainDf = trainDF.append(pd.read_csv(filesName))

    return zip(sentToTokens(sent) for sent in trainDF.original, trainDF.edit, trainDF.meanGrade)





if __name__ == '__main__':
    readTrainingData()
