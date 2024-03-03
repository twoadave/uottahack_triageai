'''TriageAI: Created for uOttaHack 6
Neural network triaging.

Created by: David J. Gayowsky, March 2nd 2024

IN THIS FILE: Actual neural network code, creation and training of neural network, and saving weights for future use.'''

#######################################################################

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################################################

#CREATE NEURAL NETWORK

#Create class for our NN:
class NeuralNetwork(nn.Module):

    #init
    def __init__(self, num_ans, nodes_1, nodes_2):
        
        super(NeuralNetwork, self).__init__()
        #Put in our layers do do doooo oh god I had like four shots at the bar i'm dying
        #helooooo whoever is reading this! :^) 
        self.layer_1 = nn.Linear(num_ans, nodes_1) 
        self.layer_2 = nn.Linear(nodes_1, nodes_2)
        #we output a single thing because whatever
        self.layer_out = nn.Linear(nodes_2, 1)
        
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(nodes_1)
        self.batchnorm2 = nn.BatchNorm1d(nodes_2)


    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.layer_out(x)
        return x