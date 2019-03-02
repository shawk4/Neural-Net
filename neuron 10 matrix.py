# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:08:16 2019

@author: Cody
"""
import data_handling
import numpy as np
import statistics
import matplotlib.pyplot as plt 

num_inputs = 2
L_rate = .05
weights = []


def Random_MLP_weights(size_of_MLP):
    for layer in range(size_of_MLP[0]):
        weights.append(np.random.uniform(-1,1,size = (size_of_MLP[1][layer] + 1, size_of_MLP[1][layer + 1])))

    
## example data input data = [[1.2, -.2], [.8, .1]], 
## Example targets input targets = [1, 0], [0, 1]
## Example size_of_MLP input size_of_MLP = [2, [num_inputs, 2, 2]] ## input number of layers then number of nodes for each layer  
def train_epoch(data, targets ,size_of_MLP):
    for i, row in enumerate(data):
        ## Lists to be cleared each iteration
        activations = []
        errors = []
        
        ## Feed forward
        activations.append(row)
        for layer in range(size_of_MLP[0]):
            ## leave bias in activations storage
            activations[layer] = np.append(activations[layer], [-1]) 
            activations.append(1 / (1 + np.exp(- np.matmul(activations[layer], weights[layer]))))
         
        ## convert weights from arrays to matricies
        for w, weight in enumerate(weights):
            weights[w] = np.matrix(weight)
        
        ## convert activations from arrays to matricies
        for a, activation in enumerate(activations):
            activations[a] = np.matrix(activation)
        
        ## calculating errors backpropogation 
        for layer in range(size_of_MLP[0]):
            l = size_of_MLP[0] - layer
            if layer == 0:
                temp = np.multiply(activations[l], 1 - activations[l])
                errors.append(np.multiply(temp, np.subtract(activations[l], targets[i])))
            else:
                temp = np.matrix(np.multiply(activations[l], 1 - activations[l]))
                temp2 = np.matmul(weights[l], np.matrix(errors[layer - 1]).T)
                temp3 = np.multiply(temp, temp2.T)
                errors.append(np.matrix(np.delete(temp3,-1,1)))        
        
        ## Updating weights
        for layer, weight in enumerate(weights):
            l = len(weights) - layer - 1
            if l == 0 :
                temp = np.matmul(np.matrix(activations[layer]).transpose(), np.matrix(errors[l]))
                weights[layer] = np.array(weight - np.multiply(L_rate, temp ))
            else:
                temp = np.matmul(np.matrix(activations[layer]).transpose(), np.matrix(errors[l])) ## np.delete(errors[l],-1,1)
                weights[layer] = np.array(weight - np.multiply(L_rate, temp ))  
    #    print('new weights')
    #    print(weights)
    #    print('')

def predict(test_data, size_of_MLP):
    outputs = []
    for i, row in enumerate(test_data):
        ## Lists to be cleared each iteration
        activations = []        
        ## Feed forward
        activations.append(row)
        
#        print('outputs by row')
#        print(outputs)
#        print('')
    
        for layer in range(size_of_MLP[0]):
            ## leave bias in activations storage
            activations[layer] = np.append(activations[layer], [-1]) 
            activations.append(1 / (1 + np.exp(- np.matmul(activations[layer], weights[layer]))))
            if layer == size_of_MLP[0] - 1:
                outputs.append(np.argmax(activations[layer + 1]))
    return outputs
    
   
## Standerdise the test data
def z_standardized(test_data):
    mean_test_data = []
    test_std = []
    for column in range(len(test_data[0])):
       mean_test_data.append(statistics.mean(test_data[:,column]))
       test_std.append(statistics.stdev(test_data[:,column]))
    for index, row in enumerate(test_data):
        test_data[index] = (row - mean_test_data) / test_std
    return test_data
    
## Calculate accuracy for each iteration
def iter_accuracy(predictions, answers, regression = False):
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics   
    return metrics.accuracy_score(answers, predictions)

## Calculate test accuracy
def accuracy(predictions, answers, regression = False):
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics   
    if (regression == True):
        print("Accuracy of k nearest neighbors via mean squared error:", metrics.mean_squared_error(answers, predictions,))
    else:     
        print("Accuracy of k nearest neighbors:", metrics.accuracy_score(answers, predictions))

def main():
    ## load in iris data
    data, targets, regression = data_handling.car()
#    print(data)
#    print(targets)
    # Train test split
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_answers, test_answers = train_test_split(data, targets, test_size = .1)
    ## is the data regressive
    regression = False

    ## count columns for input length and number of unique items in output for non regressive data
    num_attributes = data.shape[1]
    num_outputs = len(np.unique(train_answers)) ## if doing regression use 1
    
    ## create the MLP
    ## How many layers are wanted, a list of [initial_inputs, first row x nodes, second row y nodes, ... etc] 
    size_of_MLP = [2, [num_attributes, 6, num_outputs]] ## input number of layers then number of nodes for each layer
    
    ## Generate random weights for the MLP
    Random_MLP_weights(size_of_MLP)
    
#    ## standardize the data via a z score
#    test_data = z_standardized(test_data)
    train_answers_mod = []
    for answer in train_answers:
        temp = np.zeros(size_of_MLP[1][-1])
        temp[int(answer)] = 1
        train_answers_mod.append(temp)
    
    i = 0
    iteration_accuracy = []
    iteration = []
    while i < 125:
        # Run epoch
        train_epoch(train_data, train_answers_mod, size_of_MLP)
        predictions = predict(test_data, size_of_MLP) 
        iteration_accuracy.append(1 - iter_accuracy(predictions, test_answers, regression))
        iteration.append(i)
        i += 1
    ## plot the learning curve
    plt.plot(iteration, iteration_accuracy)
    plt. ylabel('accuracy')
    plt. xlabel('itterations')
    
    ## run the test
    predictions = predict(test_data, size_of_MLP)
    accuracy(predictions, test_answers, regression)
    #    print (predictions)
    
if __name__ == "__main__":
    main()       














   