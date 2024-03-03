'''TriageAI: Created for uOttaHack 6
Neural network triaging.

Created by: David J. Gayowsky, March 2nd 2024

IN THIS FILE: Actual neural network code, creation and training of neural network, and saving weights for future use.'''

#######################################################################

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

import os
import pathlib 

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################################################

#RANDOM ASS FUNCTIONS WE NEED

#Define function to add elements on to the end of a list.
def merger(seglist):
  total_list = []
  for seg in seglist:
    #Add given elements of input list to the end.
    total_list.extend(seg)
  return total_list

def get_datasets(data_set_frac):
    
    #Now we need to grab our training and testing data from file:
    script_dir = os.path.dirname(__file__)
    numbers_dir = os.path.join(script_dir, 'NN_Data/', 'data_%g_frac_heartattack.npz'%(data_set_frac))

    with np.load(numbers_dir) as f:
        poss_risk_factors = f['poss_rf']
        risk_factors_training = f['rf_training']
        poss_comb_training = f['pc_training']
        risk_factors_testing = f['rf_testing']
        poss_comb_testing = f['pc_testing']

    #Now we need to make these all into tensors and datasets...
    risk_factors_training_tensor = torch.tensor(risk_factors_training, dtype=torch.long)
    poss_comb_training_tensor = torch.tensor(poss_comb_training, dtype=torch.float32)
    training_dataset = TensorDataset(poss_comb_training_tensor, risk_factors_training_tensor)

    risk_factors_testing_tensor = torch.tensor(risk_factors_testing, dtype=torch.long)
    poss_comb_testing_tensor = torch.tensor(poss_comb_testing, dtype=torch.float32)
    testing_dataset = TensorDataset(poss_comb_testing_tensor, risk_factors_testing_tensor)

    return training_dataset, testing_dataset

#######################################################################

#CREATE NEURAL NETWORK

#Create class for our NN:
class NeuralNetwork(nn.Module):

    #init
    def __init__(self):
        
        super(NeuralNetwork, self).__init__()
        #Put in our layers do do doooo oh god I had like four shots at the bar i'm dying
        #helooooo whoever is reading this! :^) 

        self.layer_1 = nn.Linear(10, 40) 
        self.layer_2 = nn.Linear(40, 20)
        #we output a single thing because whatever
        self.layer_out = nn.Linear(20, 3)
        self.relu = nn.ReLU()
        #erm yeah do batches because this is how i did it before ok
        self.batchnorm1 = nn.BatchNorm1d(40)
        self.batchnorm2 = nn.BatchNorm1d(20)

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
    training_pass_loss =0

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

#Define a function that we use to call our neural network:
def pass_to_NN(num_epochs, batch_sze, learning_rate, training_dataset, testing_dataset):
    
    model = NeuralNetwork().to(device)
    #Declare our loss function and optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    #Create batches wwwwwwwww
    train_dataloader = DataLoader(training_dataset, batch_sze, drop_last = True, shuffle = True)
    test_dataloader = DataLoader(testing_dataset, batch_sze, drop_last = True, shuffle = True)

    #Initialize loss value.
    training_pass_loss = 0
    #Initialize array to store our loss values.
    training_losses = []

    #Make our training passes over the network.
    for i in range(1, num_epochs+1):
        #Call our training NN.
        training_step_loss = train_NN(model, loss_fn, optimizer, train_dataloader, training_pass_loss)
        #And append it to an array to store it.
        training_losses.append(training_step_loss)

    #Test our neural network.
    risk_factors_test_list, risk_factors_pred_list = test_NN(model, test_dataloader)

    #Create confusion matrix of testing and predicted values,
    results = confusion_matrix(risk_factors_test_list, risk_factors_pred_list)
    
    #Read the number of correct values off of this matrix.
    diagonal_sum = 0
    for i in range(results.shape[0]):
      diagonal_sum += results[i][i]
      
    #And then calculate our correct percentage of values.
    percentage_correct = diagonal_sum/len(risk_factors_test_list) * 100
    
    return percentage_correct, training_losses

#######################################################################

if __name__ == '__main__':

    condition = 'heart attack'
    data_set_frac = 0.5
    num_epochs = 25
    batch_sze = 20
    learning_rate = 0.0003

    training_dataset, testing_dataset = get_datasets(data_set_frac)

    percentage_correct, training_losses = pass_to_NN(num_epochs, batch_sze, learning_rate, training_dataset, testing_dataset)

    print(percentage_correct)