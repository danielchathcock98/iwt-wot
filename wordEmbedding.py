import fasttext
from pathlib import Path

MODEL_PATH = Path('englishVecs/cc.en.300.bin')

embeddingModel = None

def prepare_sequence(sentence):
    if embeddingModel is None:
        embeddingModel = fasttext.load_model(MODEL_PATH)

    return [embeddingModel[word] for word in sentence]
