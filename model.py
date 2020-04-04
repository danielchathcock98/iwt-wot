import torch.nn as nn
import torch.nn.functional as F

class Model2(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BasicPOSTagger, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # You are required to create a model with:
        # an embedding layer: that maps words to the embedding space
        # an LSTM layer: that takes word embeddings as input and outputs hidden states
        # a Linear layer: maps from hidden state space to tag space
        #############################################################################
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=LSTM_LAYERS, dropout=DROPOUT)
        self.decoder = nn.Linear(hidden_dim, tagset_size)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, sentence):
        tag_scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        # Given a tokenized index-mapped sentence as the argument,
        # compute the corresponding scores for tags
        # returns:: tag_scores (Tensor)
        #############################################################################

        encoded = self.encoder(sentence)
        lstmOut, _ = self.lstm(encoded.view(-1, 1, self.embedding_dim))
        tag_scores = self.decoder(lstmOut.view(-1, self.hidden_dim))

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return tag_scores
