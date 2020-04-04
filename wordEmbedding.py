import fasttext
from pathlib import Path

MODEL_PATH = Path('englishVecs/cc.en.300.bin')

class Embedding():

    def __init__(self):
        print('loading embedding model')
        self.embeddingModel = fasttext.load_model(MODEL_PATH)
        print('loaded embedding model')


    def embed_sentence(self, sentence):
        return [self.embeddingModel[word] for word in sentence]
