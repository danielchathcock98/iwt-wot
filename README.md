# iwt-wot

Project by Sherry Sarkar and Daniel Hathcock

## Directory Structure

All Python files are in the top level directory (iwt-wot). The main file to run models is the Jupyter notebook iwt-wot.ipynb. Other files are utilities to preprocess data, define models, etc., and are mostly called within the Jupyter notebook. The baselines.py and makeFigures.py files can be run in a standalone manner to generate the baseline scores and some of the figures in our final report.

## How to run

### Required packages

jupyter
pytorch
numpy
scipy
pandas
matplotlib
fasttext*

*To install fasttext, you must build from source in order to get access to the pretrained vectors. To do so, please follow instructions at [https://fasttext.cc/docs/en/support.html#building-fasttext-python-module](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module).

### Retrieve data

After installing fasttext, open a Python interpretter and run

```
>>> import fasttext.util
>>> fasttext.util.download_model('en', if_exists='ignore')  # English
```

Make a directory in the iwt-wot directory titled `englishVecs`, and place the file `cc.en.300.bin` downloaded by fasttext into this directory.

### Run code

Can run baselines.py and makeFigures.py independently. To run iwt-wot.ipynb, open it in a Jupyter notebook. Ignore the first 4 cells (only necessary if running in Google Colab). In 5th cell, change DRIVE_DIR path to reflect the location of the iwt-wot directory (should be the working directory, unless you moved iwt-wot.ipynb). Then all other cells can be run as usual. 