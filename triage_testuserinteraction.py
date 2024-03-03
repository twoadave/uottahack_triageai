'''TriageAI: Created for uOttaHack 6
Neural network triaging.

Created by: David J. Gayowsky, March 2nd 2024

IN THIS FILE: Pretend we're a user submitting something, in the way that I guess a backend dev can do.'''

#######################################################################

import numpy as np

import os
import pathlib 

from triage_neuralnet import NeuralNetwork
from triage_interactionnet import import_user_data, import_model, test_user_data

#######################################################################

#Function to generate some random user data:
def generate_test_user_data(num_tests, num_ans):

    np.random.seed()

    #Shit for saving
    basefolder = pathlib.Path(__file__)

    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'NN_Data/')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    #Generate random ass data
    for i in range(num_tests):
        rand_condition_val = np.random.randint(2)
        if rand_condition_val == 0:
            rand_condition = 'heart_attack'
        else:
            rand_condition = 'wound'
        rand_ans = np.random.randint(2,size=num_ans)

        np.savez_compressed(data_dir + 'data_%g_user_test.npz'%(i), user_data = rand_ans, user_condition = rand_condition)

#Define function to load test user data.
def load_test_user_data(i):

    #Now we need to grab our training and testing data from file:
    script_dir = os.path.dirname(__file__)
    filepath = os.path.join(script_dir, 'NN_Data/', 'data_%g_user_test.npz'%(i))

    condition, user_data = import_user_data(filepath)

    return condition, user_data

    
#######################################################################
    
if __name__ == '__main__':

    generate_test_user_data(1, 10)
    condition, user_data = load_test_user_data(0)
    model = import_model(condition)
    user_risk_factor = test_user_data(model, user_data)

    print('Given the condition: ' + str(condition) + ', the user risk factor is assessed to be: ' + str(user_risk_factor))


