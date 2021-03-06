import fasttext

MODEL_PATH = 'englishVecs/cc.en.300.bin'

class Embedding():

    def __init__(self, path_prepend):
        print('loading embedding model')
        self.embeddingModel = fasttext.load_model(str(path_prepend / MODEL_PATH))
        print('loaded embedding model')


    def embed_sentence(self, sentence):
        return [self.embeddingModel[word] for word in sentence]
