import torch

import torch.optim as optim
import random

from model import Model2
from preprocessing import model2Preprocessing

EMBEDDING_DIM = 8
HIDDEN_DIM = 12
LEARNING_RATE = 0.3
LSTM_LAYERS = 1
DROPOUT = 0
EPOCHS = 30

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

        sent_seq = prepare_sequence(sentence, word_to_idx).to(device)
        tag_seq = prepare_sequence(tags, tag_to_idx).to(device)

        tag_output = model(sent_seq)

        loss = loss_function(tag_output, tag_seq)

        train_loss += loss.item()
        train_examples += len(sentence)

        loss.backward()
        optimizer.step()

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_accuracy = evaluate(model, loss_function, optimizer)

    print("Epoch: {}/{}\tAvg Train Loss: {:.4f}\tAvg Val Loss: {:.4f}\t Val Accuracy: {:.0f}".format(epoch,
                                                                      EPOCHS,
                                                                      avg_train_loss,
                                                                      avg_val_loss,
                                                                      val_accuracy))

def evaluate(model, loss_function, optimizer):
  # returns:: avg_val_loss (float)
  # returns:: val_accuracy (float)
    val_loss = 0
    correct = 0
    val_examples = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            #############################################################################
            # TODO: Implement the evaluate loop
            # Find the average validation loss along with the validation accuracy.
            # Hint: To find the accuracy, argmax of tag predictions can be used.
            #############################################################################

            sent_seq = prepare_sequence(sentence, word_to_idx).to(device)
            tag_seq = prepare_sequence(tags, tag_to_idx).to(device)

            tag_output = model(sent_seq)

            loss = loss_function(tag_output, tag_seq)
            val_loss += loss.item()
            correct += (tag_output.argmax(1) == tag_seq).sum().item()

            val_examples += len(sentence)

            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
    val_accuracy = 100. * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy




if __name__ == '__main__':
    random.seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BasicPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, loss_function, optimizer)
