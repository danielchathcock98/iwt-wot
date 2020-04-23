import torch.nn as nn
import torch.nn.functional as F
import torch

EMBEDDING_DIM = 300

class Model25(nn.Module):

    def __init__(self, hidden_dim, lstm_layers, dropout):
        super(Model25, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim, num_layers=lstm_layers, dropout=dropout)

        # We output 2D vector (binary classification).
        self.decoder = nn.Linear(hidden_dim, 2)  
        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, encoded_sentence):

        _, (lstm_hn, _) = self.lstm(encoded_sentence.view(-1, 1, EMBEDDING_DIM))
        decoded = self.decoder(lstm_hn.view(self.hidden_dim))
        probability_vector = self.softmax_layer(decoded)

        # Return number between 0 and 3
        return 3 * probability_vector[0]

class Model3(nn.Module):

    def __init__(self, hidden_dim, lstm_layers, dropout):
        super(Model3, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTM(EMBEDDING_DIM, hidden_dim, num_layers=lstm_layers, dropout=dropout)
        self.lstm2 = nn.LSTM(EMBEDDING_DIM, hidden_dim, num_layers=lstm_layers, dropout=dropout)

        # We pass in 2 lstm layers, concatenated. We output 2D vector (binary classification).
        self.decoder = nn.Linear(2*hidden_dim, 2)  
        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, encoded_sentence1, encoded_sentence2):

        _, (lstm_hn1, _) = self.lstm1(encoded_sentence1.view(-1, 1, EMBEDDING_DIM))
        _, (lstm_hn2, _) = self.lstm2(encoded_sentence2.view(-1, 1, EMBEDDING_DIM))

        # Concatenating might be done wrong.
        decoded = self.decoder(torch.cat((lstm_hn1.view(self.hidden_dim), lstm_hn2.view(self.hidden_dim)), 0)) 
        probability_vector = self.softmax_layer(decoded)

        # Return number between 0 and 3
        return 3 * probability_vector[0]
