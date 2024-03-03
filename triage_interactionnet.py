'''TriageAI: Created for uOttaHack 6
Neural network triaging.

Created by: David J. Gayowsky, March 2nd 2024

IN THIS FILE: Import of data from user, loading of pre-trained neural network and return of assessed risk value.'''

#######################################################################

import torch
import os
import pathlib 
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

from triage_neuralnet import NeuralNetwork

#######################################################################

#don't TOUCH
data_set_frac = 0.5
num_epochs = 50
batch_sze = 20
learning_rate = 0.0003

#######################################################################

#Define a function to retrieve our user data and condition from somewhere:
def import_user_data(filepath):

    with np.load(filepath) as f:
        user_ans = f['user_data']
        condition = f['user_condition']

    user_ans = np.array([np.array(user_ans)])

    given_user_data = torch.tensor(user_ans, dtype=torch.float32)
    risk_factor_default = torch.tensor([1], dtype=torch.long)

    user_data_tensor = TensorDataset(given_user_data, risk_factor_default)
    user_data = DataLoader(user_data_tensor, 1, drop_last = False, shuffle = False)

    return condition, user_data

#Define a function to import our neural network model:
def import_model(condition):

    script_dir = os.path.dirname(__file__)
    if condition == 'heart_attack':
       model_dir = os.path.join(script_dir, 'NN_Data/', 'heart attackmodel_complete.pth')
    else:
       model_dir = os.path.join(script_dir, 'NN_Data/', 'woundmodel_complete.pth')

    print(model_dir)

    model = torch.load(model_dir)

    return model

#Define a function to classify our user data into a risk category:
def test_user_data(model, user_data):

    model = model.to(device)

    with torch.no_grad():
        #Put our network into testing mode:
        model.eval()

        for combination_batch, risk_factors_batch in user_data:
            combination_batch = combination_batch.to(device)
            user_risk_factor_pred = model(combination_batch.float())
            _, risk_factor_pred_tag = torch.max(user_risk_factor_pred, dim = 1)
            user_risk_factor = risk_factor_pred_tag.cpu().numpy()

    return user_risk_factor[0]

#######################################################################

if __name__ == '__main__':

    condition, user_data = import_user_data()
    model = import_model(condition)

    user_risk_factor = test_user_data(model, user_data)