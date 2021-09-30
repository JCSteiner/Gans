###############################################################################
# Embedding_Trainer.py
# J. Steiner

#%%## LOADS DEPENDENCIES

# imports pytorch functionality
import torch
import torch.nn as nn
import torch.functional as f
from torch.nn.modules.loss import NLLLoss
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

#%%## N GRAM EMBEDDING MODEL
# a class to train an embedding dict according to the n-gram model
class NGramEmbedding(nn.Module):

    # defines the hidden dimensions of the model as an internal parameter
    HIDDEN_DIMS = 1024

    # class constructor
    def __init__(self, vocab_size, embedding_dims, context_size):
        
        # vocab_size - the number of embeddins we are going to make
        # embedding_dims - the number of dimensions we are going to embed into
        # context_size - the number of future words we consider in predicting
        #                the next word

        # runs the base class constructor
        super(NGramEmbedding, self).__init__()

        # defines the embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dims)

        # defines the hidden layer with relu activation
        self.hidden = nn.Sequential( nn.Linear(context_size * embedding_dims, self.HIDDEN_DIMS),
                                     nn.ReLU() )

        # defines the output layer 
        self.output = nn.Sequential( nn.Linear( self.HIDDEN_DIMS, vocab_size ), nn.LogSoftmax(dim=1) )

    # the forward pass thorugh the network
    def forward(self, x):

        # x - the input into the network

        # embeds the input vector
        x = self.embeddings(x)
        # casts to the hidden layer
        x = self.hidden(x.view(x.shape[0], x.shape[1] * x.shape[2]))
        # casts back to the output size
        x = self.output(x)

        # returns the model output
        return x

    # forward pass through only the embedding layer
    def embed(self, x):
        with torch.no_grad():
            # embeds the input vector
            return self.embeddings(x)
#%%## DATA PROCESSING FUNCTION

# creates word grams from the n-gram model
def create_grams(sequences):

    # sequences - the sequences we want to encode using the n-gram model
    
    # initializes an empty list to store the grams
    grams = []

    # loops through each sequence in the list of sequences
    for x, y in sequences:
        
        seq = x #+y

        # creates according to a 3 gram model
        gram = [ (torch.tensor([ seq[i-1], seq[i], seq[i+1] ]), torch.tensor([ seq[i+2] ])) for i in range(1, len(seq)-2) ]

        # appends the gram to the gram list
        grams += gram

    # converts to a tensor and returns
    return grams

#%%## EMBEDDING TRAINER FUNCTION

# trains the embeddings based on the sequences passed in
def train_embeddings(sequences, epochs, learning_rate, batch_size, vocab_size, embedding_dims, context_size, verbose = True):

    # sequences - a list of sequences to embed on
    # epochs - how many epochs to train for
    # learning_rate - the learning rate during training
    # vocab_size - the number of words we are going to embed
    # embedding_dims - the dimensionality of our embeddingsw
    # context_size - how many words we consider at once when training

    # creates an instance of the n-gram embedding model
    model = NGramEmbedding(vocab_size, embedding_dims, context_size)

    # creates a loss function object
    loss_func = nn.NLLLoss()
    # creates an SDG optimizer model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    if verbose:
        print('Creating Grams...')
    
    # encodes the training set with the n-gram model
    grams = create_grams(sequences)

    training_data = DataLoader(grams, batch_size = batch_size, shuffle = True)
    if torch.cuda.is_available():
        training_data.cuda()

    if verbose:
        print('Grams Created.')

    # if it is possible to train using the gpu, does that
    if torch.cuda.is_available():
        model.cuda()

    if verbose:
        print('Training...')

    # loops through each epoch in the range of epochs
    for epoch in range(epochs):

        # zeros out the cost
        cost = 0

        batch_ctr = 0

        # for each input and output in the training set
        for x, y in iter(training_data):

            # zeros out model gradients
            model.zero_grad()

            # predicts the next word based on the input sequence x
            pred = model(x)

            # calculates the loss based on the predicted and actual next word
            loss = loss_func(pred.view(pred.shape[0], pred.shape[1], 1), y)

            # backwards pass through the network and increments
            loss.backward()
            optimizer.step()

            # increments the cost
            cost += loss.item()
            batch_ctr += 1

        # steps the learning rate scheduler
        scheduler.step()
        
        # prints status
        if verbose:
            print('Epoch:', epoch + 1, 'Cost:', cost / len(sequences))
        # every 10 epochs, save the model parameters
        if epoch == epochs - 1 or epoch % 10 == 0 :
            torch.save(model.state_dict(), './Embedding_States/embedder_epoch_' + str(epoch))

    # if we are in verbose mode prints a status update
    if verbose:
        print('Training Complete.')

# from DataLoader import load_prompts
# data, word2Idx, idx2Word = load_prompts('./Data/writingPrompts/valid_wp_source.txt', './Data/writingPrompts/valid_wp_target.txt')
# train_embeddings(data, 200, 1e-1, 100, len(word2Idx), 256, 3)
