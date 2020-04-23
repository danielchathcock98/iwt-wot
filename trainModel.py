import torch

import torch.optim as optim
import torch.nn as nn
import random
import math

from model import Model2, LEARNING_RATE, EPOCHS
from preprocessing import model2preprocessing, TASK_1, EXTRA_TRAIN_TASK_1
from wordEmbedding import Embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = None
val_data = None
embed = None

def train(epoch, model, loss_function, optimizer):
    train_loss = 0
    train_examples = 0

    for inputData, score in training_data:
        #############################################################################
        # TODO: Implement the training loop
        # Hint: you can use the prepare_sequence method for creating index mappings
        # for sentences. Find the gradient with respect to the loss and update the
        # model parameters using the optimizer.
        #############################################################################
        model.zero_grad()

        embedding = torch.tensor(embed.embed_sentence(inputData)).to(device)
        #score_tensor = torch.tensor([score]).to(device)    OLD WAY OF USING SCORE

        thresholded_score = round(score/3)
        score_tensor = torch.tensor([thresholded_score]).to(device)

        score_output = model(embedding)

        loss = loss_function(score_output, score_tensor)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        train_examples += 1

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_rmse = evaluate(model, loss_function, optimizer)

    print("Epoch: {}/{}\tAvg Train Loss: {:.4f}\tAvg Val Loss: {:.4f}\t Val RMSE: {:.4f}".format(epoch,
                                                                      EPOCHS,
                                                                      avg_train_loss,
                                                                      avg_val_loss,
                                                                      val_rmse))

def evaluate(model, loss_function, optimizer):
  # returns:: avg_val_loss (float)
  # returns:: val_accuracy (float)
    val_loss = 0
    square_error = 0
    val_examples = 0
    with torch.no_grad():
        for inputData, score in val_data:
            #############################################################################
            # TODO: Implement the training loop
            # Hint: you can use the prepare_sequence method for creating index mappings
            # for sentences. Find the gradient with respect to the loss and update the
            # model parameters using the optimizer.
            #############################################################################
            embedding = torch.tensor(embed.embed_sentence(inputData)).to(device)
            #score_tensor = torch.tensor([score]).to(device)  OLD
            thresholded_score = round(score / 3)
            score_tensor = torch.tensor([thresholded_score]).to(device)

            score_output = model(embedding)

            loss = loss_function(score_output, score_tensor)
            val_loss += loss.item()
            square_error += (3*math.exp(score_output[1].item()) - score)**2

            val_examples += 1

            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################

    rmse = math.sqrt(square_error / val_examples)
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, rmse



if __name__ == '__main__':
    random.seed(1)
    print(device)

    model = Model2().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.NLLLoss()

    training_data = model2preprocessing([TASK_1 / 'train.csv', EXTRA_TRAIN_TASK_1])
    val_data = model2preprocessing([TASK_1 / 'dev.csv'])
    embed = Embedding()

    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, loss_function, optimizer)
