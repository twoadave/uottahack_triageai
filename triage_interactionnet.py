'''TriageAI: Created for uOttaHack 6
Neural network triaging.

Created by: David J. Gayowsky, March 2nd 2024

IN THIS FILE: Import of data from user, loading of pre-trained neural network and return of assessed risk value.'''

#######################################################################

import torch
import os
import pathlib 
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################################################

#don't TOUCH
data_set_frac = 0.5
num_epochs = 50
batch_sze = 20
learning_rate = 0.0003

#######################################################################

#Define a function to retrieve our user data and condition from somewhere:
def import_user_data():
    #Note: Import condition from user.
    condition = 'wound'
    #Note: Import user_data from user.
    user_data = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1])

    return condition, user_data

#Define a function to import our neural network model:
def import_model(condition):

    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, 'NN_Data/', condition, 'model_complete.pth')

    model = torch.load(model_dir)

    return model

#Define a function to classify our user data into a risk category:
def test_user_data(model, user_data):

    #Make sure to put it into eval mode!
    model.eval()

    with torch.no_grad():
        user_risk_factor = model(user_data)

    return user_risk_factor

#######################################################################

if __name__ == '__main__':

    condition, user_data = import_user_data()
    model = import_model(condition)

    user_risk_factor = test_user_data(model, user_data)

    print(user_risk_factor)