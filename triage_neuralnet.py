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

#Define function to add elements on to the end of a list.
def merger(seglist):
  total_list = []
  for seg in seglist:
    #Add given elements of input list to the end.
    total_list.extend(seg)
  return total_list

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
        #erm yeah do batches because this is how i did it before ok
        self.batchnorm1 = nn.BatchNorm1d(nodes_1)
        self.batchnorm2 = nn.BatchNorm1d(nodes_2)

    #define forward pass
    def forward(self, inputs):
        #inputs! relu! batches! outputs! they're all here!
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.layer_out(x)
        return x
    
#######################################################################
    
#TRAINING AND TESTING NEURAL NETWORK
    
#Define a function to train our neural network whee
def train_NN(model, loss_fn, optimizer, training_dataloader, training_loss):

    #Training wheels on children:
    model.train()

    #For each batch in our training data:
    for poss_comb_batch, risk_factors_batch in training_dataloader:
        #Send a batch of possible answer configurations to our network:
        poss_comb_batch = poss_comb_batch.to(device)
        risk_factors_batch = risk_factors_batch.type(torch.LongTensor)
        risk_factors_batch = risk_factors_batch.to(device)

        #Effectively forward pass lol
        optimizer.zero_grad()
        #compute our actual predicted risk value:
        risk_pred = model(poss_comb_batch.float())

        #Compute loss:
        train_loss = loss_fn(risk_pred, risk_factors_batch)

        #Backwards pass:
        train_loss.backward()
        optimizer.step()
        training_pass_loss += train_loss.item()

    return training_pass_loss

#Define a function which we can call to test our neural network:
def test_NN(model, test_dataloader):
    
    #Initialize list of testing and predicted risk factors:
    risk_factors_test_list = []
    risk_factors_pred_list = []

    with torch.no_grad():
      #Put our network into testing mode:
      model.eval()
      
      #For all:
      for poss_comb_batch, risk_factors_batch in test_dataloader:
        poss_comb_batch = poss_comb_batch.to(device)
        risk_factors_test_pred = model(poss_comb_batch.float())
        _, risk_factors_pred_tags = torch.max(risk_factors_test_pred, dim = 1)
        risk_factors_pred_list.append(risk_factors_pred_tags.cpu().numpy())
        risk_factors_test_list.append(risk_factors_batch)
    
    if len(risk_factors_pred_list) > 1:
        risk_factors_pred_list = [a.squeeze().tolist() for a in risk_factors_pred_list]
        risk_factors_test_list = [a.squeeze().tolist() for a in risk_factors_test_list]
        #Well we throw all these lists together here.
        risk_factors_test_list = merger(risk_factors_test_list)
        risk_factors_pred_list = merger(risk_factors_pred_list)
    
    return risk_factors_test_list, risk_factors_pred_list