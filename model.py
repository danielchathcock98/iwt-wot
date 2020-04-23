import torch.nn as nn
import torch.nn.functional as F
import torch

EMBEDDING_DIM = 300
HIDDEN_DIM = 10
LEARNING_RATE = 1
LSTM_LAYERS = 1
DROPOUT = 0
EPOCHS = 30

class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.hidden_dim = HIDDEN_DIM

        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=LSTM_LAYERS, dropout=DROPOUT)
        self.decoder = nn.Linear(HIDDEN_DIM, 2)  # We output 2D vector (binary classification).
        self.softmax_layer = nn.LogSoftmax(dim=0)

    def forward(self, encoded_sentence):

        _, (lstm_hn, _) = self.lstm(encoded_sentence.view(-1, 1, EMBEDDING_DIM))
        decoded = self.decoder(lstm_hn.view(self.hidden_dim))
        probability_vector = self.softmax_layer(decoded)

        return probability_vector

class Model25(nn.Module):

    def __init__(self):
        super(Model25, self).__init__()
        self.hidden_dim = HIDDEN_DIM

        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=LSTM_LAYERS, dropout=DROPOUT)
        self.decoder = nn.Linear(HIDDEN_DIM, 2)  # We output 2D vector (binary classification).
        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, encoded_sentence):

        _, (lstm_hn, _) = self.lstm(encoded_sentence.view(-1, 1, EMBEDDING_DIM))
        decoded = self.decoder(lstm_hn.view(self.hidden_dim))
        probability_vector = self.softmax_layer(decoded)

        # Return number between 0 and 3
        return 3 * probability_vector[0]

class Model3(nn.Module):

    def __init__(self):
        super(Model3, self).__init__()
        self.hidden_dim = HIDDEN_DIM

        self.lstm1 = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=LSTM_LAYERS, dropout=DROPOUT)
        self.lstm2 = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=LSTM_LAYERS, dropout=DROPOUT)
        self.decoder = nn.Linear(2*HIDDEN_DIM, 2)  # We pass in 2 lstm layers, concatenated. We output 2D vector (binary classification).
        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, encoded_sentence):

        _, (lstm_hn1, _) = self.lstm1(encoded_sentence.view(-1, 1, EMBEDDING_DIM))
        _, (lstm_hn2, _) = self.lstm2(encoded_sentence.view(-1, 1, EMBEDDING_DIM))

        decoded = self.decoder(torch.cat((lstm_hn1.view(self.hidden_dim), lstm_hn2.view(self.hidden_dim)), 0))  # Concatenating might be done wrong.
        probability_vector = self.softmax_layer(decoded)

        # Return number between 0 and 3
        return 3 * probability_vector[0]
