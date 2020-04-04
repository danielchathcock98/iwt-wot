import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
HIDDEN_DIM = 4
LEARNING_RATE = 0.2
LSTM_LAYERS = 1
DROPOUT = 0
EPOCHS = 30

class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # You are required to create a model with:
        # an embedding layer: that maps words to the embedding space
        # an LSTM layer: that takes word embeddings as input and outputs hidden states
        # a Linear layer: maps from hidden state space to tag space
        #############################################################################
        # self.embedding_dim = embedding_dim
        self.hidden_dim = HIDDEN_DIM

        #self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=LSTM_LAYERS, dropout=DROPOUT)
        self.decoder = nn.Linear(HIDDEN_DIM, 1)  # We output a single score now.

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, encoded_sentence):
        tag_scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        # Given a tokenized index-mapped sentence as the argument,
        # compute the corresponding scores for tags
        # returns:: score (Tensor)
        #############################################################################

        #encoded = self.encoder(sentence)
        #lstmOut, _ = self.lstm(encoded.view(-1, 1, self.embedding_dim))
        lstmOut, _ = self.lstm(encoded_sentence.view(-1, 1, self.embedding_dim))
        score = self.decoder(lstmOut.view(-1, self.hidden_dim))

        return score
