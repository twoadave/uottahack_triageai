'''TriageAI: Created for uOttaHack 6
Neural network triaging.

Created by: David J. Gayowsky, March 2nd 2024'''

#######################################################################

import numpy as np
import itertools

import matplotlib.pyplot as plt
import os
import pathlib 

import torch
from torch.utils.data import TensorDataset

#######################################################################

'''Create test case criteria: 
    Heart attack,
    Paper cut.
    Where each scenario has ans = 10 yes or no questions, e.g. becomes an array of 10 binary values.
    2^10 possible scenarios per situation!

Create test case data sets:
    Take all combinations without repetition of criteria, and calculate corresponding risk_categories = 3 for risk level (0, 1, 2).
    Compute the amount of each risk level data present (e.g. number of data points corresponding to each category).
    Split N data points into training and testing data sets, where the test data set consists of M total data points, where we make sure to sample from smaller parts of the distribution e.g. edge cases. Such that num_risk_i ≈ num_risk_j = M/risk_categories for all i ≠ j wherever possible.
    Note: is there a way to do this where we can expand to risk_categories cases? Look up efficient data sampling? 

Train and test neural network using subsets of training data of order P = Q/M ≤ M, measuring speed and accuracy each time, finding a balance between accuracy and speed. 
    More training data = more accuracy, but lower speed!
    Use timer to measure how much time each run takes.
    Additionally, look at effect of number of nodes/layers on speed and accuracy.
    Find order of operations?
    Note: is there an ideal level of neural network accuracy? Say, for healthcare based applications, do we want our neural network to be accurate 99.5% of the time, or is 95% sufficient?

Save neural network weights with desired accuracy.

Test again using saved neural network weights, loading from file and measuring time.

Demo test cases.
'''

#######################################################################

#CREATION OF DATA

#Define a function to create a sample answerspace:
def create_sample_answerspace(num_ans):

    #Create all possible lists of combinations without repetition that are N in length of 1 or 0:
    poss_comb = list(itertools.product(range(2), repeat=num_ans))
    #Convert to array for use in neural network.
    poss_comb = np.array(poss_comb)

    return poss_comb

#Define a function to calculate overall risk factors, weighted by symptom.
def calculate_sample_risk_factors(poss_comb, condition, num_ans):

    #Note: Here we explicitly define symptoms and weight them according to patient's assumed condition.
    #Using standard weighted average: sum of terms divided by number of terms.

    all_risk_factors = []

    #Define weight vector associated with questions with condition:
    if condition == 'heart attack':
        weight_vector = [2, 1, 1, 1, 1, 3, 3, 2, 3, 2]
    else:
        weight_vector = [2, 1, 1, 2, 2, 1, 1, 2, 1, 1]

    #Calculate risk factor associated with each possible combination of answers:
    for i in range(len(poss_comb)):
        risk_factor = np.dot(poss_comb[i], weight_vector)/num_ans
        if risk_factor < 1:
            risk_factor = np.round(risk_factor)
        else:
            risk_factor = np.ceil(risk_factor)
        all_risk_factors.append(risk_factor)

    poss_risk_factors = [0, 1, 2]
    total_num_datapoints = len(all_risk_factors)

    return all_risk_factors, poss_risk_factors, total_num_datapoints

#Define a function to look at distribution of our data via histogram.
def distribution_view(all_risk_factors):

    basefolder = pathlib.Path(__file__)

    script_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(script_dir, 'Plots/')

    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    plt.hist(all_risk_factors)
    plt.title('Distribution of Calculated Risk Factors for Heart Attack \n Across Sample Answerspace')
    plt.xlabel('Risk Factor Value')
    plt.ylabel('Number of Possibled Responses')
    plt.savefig(plots_dir + 'answerspace_riskfactor_dist_rounded.png')

#######################################################################
    
#CREATION OF TRAINING AND TESTING DATASETS

#Throw our data into tensors that we can use.
'''def create_tensor(poss_comb, all_risk_factors):

    all_responses_tensor = torch.tensor(poss_comb)
    all_risk_factors_tensor = torch.tensor(all_risk_factors)

    return all_responses_tensor, all_risk_factors_tensor'''

#Now create a function to split into training and testing data.
def create_tt_data(all_risk_factors, poss_comb, total_num_datapoints, data_set_frac):

    #We want to do this probabilistically, so we want approximately the same 
    #amount of data from each part of the sample space. 
    #We also want to use minimal training and testing data, so want to declare the amount of data points we'd like to use.

    #Data set frac represents the fraction of the total data set we would like to use for training
    #and testing, where 0 < data_set_frac <= 0.5.

    #For M total training data points, we say:
    data_points_per_set = np.floor(data_set_frac*total_num_datapoints)

    #Create a shuffled random list of indices we want to grab:
    shuffled_indices = np.arange(total_num_datapoints)
    np.random.shuffle(shuffled_indices)

    #Grab a fraction of these indices to be our training data, randomly sampled:
    training_indices = shuffled_indices[:data_points_per_set]

    #Now, get a list of our risk factors as a function of these indices.
    risk_factors_training = all_risk_factors[training_indices]
    poss_comb_training = poss_comb[training_indices]

    #Convert to a tensor:
    risk_factors_training_tensor = torch.tensor(risk_factors_training)
    poss_comb_training_tensor = torch.tensor(poss_comb_training)
    training_dataset = TensorDataset(poss_comb_training_tensor, risk_factors_training_tensor)

    #Now repeat the above but with testing data:
    testing_indices = shuffled_indices[data_points_per_set:]
    risk_factors_testing = all_risk_factors[testing_indices]
    poss_comb_testing = poss_comb[testing_indices]

    #Convert to a tensor:
    risk_factors_testing_tensor = torch.tensor(risk_factors_testing)
    poss_comb_testing_tensor = torch.tensor(poss_comb_testing)
    testing_dataset = TensorDataset(poss_comb_testing_tensor, risk_factors_testing_tensor)

    return training_dataset, testing_dataset

    #An attempt at random over/under sampling - depreciated because it's stupid and also wasn't really working lol
    #there are like, better ways to do this so the intent was there
    '''#Figure out the indices of the classes:
    risk_0_indices = np.where(all_risk_factors == 0)
    risk_1_indices = np.where(all_risk_factors == 1)
    risk_2_indices = np.where(all_risk_factors == 2)

    #Now shuffle these:
    np.random.shuffle(risk_0_indices)
    np.random.shuffle(risk_1_indices)
    np.random.shuffle(risk_2_indices)

    #Now take the bottom number of data points per class to use as our training data:
    training_0_indices = risk_0_indices[:data_points_per_class]
    training_1_indices = risk_1_indices[:data_points_per_class]
    training_2_indices = risk_2_indices[:data_points_per_class]

    #Now we have the training combinations, and we know their associated risk factors...
    training_0_combs = poss_comb[training_0_indices]
    training_1_combs = poss_comb[training_1_indices]
    training_2_combs = poss_comb[training_2_indices]'''





    

#######################################################################
    
if __name__ == '__main__':

    num_ans = 10
    poss_comb = create_sample_answerspace(num_ans)
    all_risk_factors, poss_risk_factors = calculate_sample_risk_factors(poss_comb, 'heart attack', num_ans)

    distribution_view(all_risk_factors)