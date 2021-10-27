# Import packages

import random as python_random
import numpy as np
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

# Functions

def dataset_generator(s,k,num_pos_samples,num_neg_samples, loc_res, scale_res,
                      set_seed):
    
    """Creates an instance of a multilayer perceptron (MLP) neural network.

    # Parameters
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        X:    
        y: 
    
    """
    
    np.random.seed(set_seed) 
    negative_samples = np.random.normal(loc = loc_res, scale = scale_res,  
                                    size = (num_neg_samples * k,s))   
    negative_samples = negative_samples.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() 
    overall_mean = negative_samples.mean(axis=1) 
    sample_range = negative_samples.max(axis=1) - negative_samples.min(axis=1) 
    negative_samples = np.c_[negative_samples,overall_mean,sample_range] 
    
    shift = [1,2,3] 
    positive_samples_all = np.zeros((1, 8))
    
    for i in shift:
        for j in range(s-1): # number of OC streams
            for l in combinations([0,1,2,3,4,5], j + 1):
                np.random.seed(0) # seed 
                positive_samples = np.random.normal(loc = loc_res, scale = scale_res, size = (num_pos_samples * k,s))
                positive_samples[:, np.array(l)] = positive_samples[:, np.array(l)] + i
                positive_samples = positive_samples.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() 
                overall_mean = positive_samples.mean(axis=1) 
                sample_range = positive_samples.max(axis=1) - positive_samples.min(axis=1) 
                positive_samples = np.c_[positive_samples,overall_mean,sample_range]
                positive_samples_all = np.vstack([positive_samples_all, positive_samples])
    positive_samples_all = np.delete(positive_samples_all, (0), axis=0) # remove the firts row
    
    X = np.vstack([negative_samples, positive_samples_all])
    y = np.repeat([0,1], X.shape[0]/2)
    
    return X,y
    

def ROC_AUC_plot(classifier, X_val, y_val, f, xlabel = '', ylabel = '', legend_fontsize = 10, label_fontsize = 12, 
                text_fontsize = 10, text_position = [0.2,0.7], tick_labelsize = 10):
         
    y_prob = classifier.predict(X_val)

    fpr, tpr, threshold = roc_curve(y_val, y_prob)
    auc_val = auc(fpr, tpr)

    plt.plot(fpr, tpr, linestyle='-', color = "blue", label = "ROC curve")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label = "Random reference line")
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)

    plt.text(text_position[0], text_position[1], 'AUC = '+ str(round(auc_val,3))[0:4], fontsize=text_fontsize, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = tick_labelsize)

    return f

def set_cv_alpha(n, s, k, loc_res, scale_res ,set_seed, scaler, classifier, cv):

    np.random.seed(set_seed)
    obs = np.random.normal(loc = loc_res, scale = scale_res, size = (n * k, s))
    obs = obs.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose()
    mean = obs.mean(axis=1)
    r = obs.max(axis=1) - obs.min(axis=1)
    vec = np.c_[obs,mean,r]
    vec = scaler.transform(vec)
    y_pred = classifier.predict(vec)
    y_pred = (y_pred>cv)*1
    alpha = (len(y_pred[y_pred == 1])/n)

    return alpha


def NN_model(hidden_activation_function, num_hidden_layer, num_hidden_neuron):
    """Creates an instance of a multilayer perceptron (MLP) neural network.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model.
    """
    
    # To obtain reproducible result susing Keras during development
    np.random.seed(2)
    python_random.seed(123) 
    tf.random.set_seed(1234) 
    
    output_activation = 'sigmoid'
    output_units = 1
    
    classifier = Sequential()
    
    for i in range(num_hidden_layer):
        classifier.add(Dense(units=num_hidden_neuron[i], 
                             activation=hidden_activation_function[i]))
    
    classifier.add(Dense(units=output_units, activation=output_activation))

    return classifier


def control_chart(NN_pred, fig_control_chart, CV, xlabel = "", ylabel = "",
                  text_position = [35,0.9], text_fontsize = 12, label_fontsize = 12,
                  tick_labelsize = 10):
    
    x = np.arange(2,len(NN_pred) + 2,1)

    plt.plot(x, NN_pred, color='black', ls='-', marker='*')
    plt.axhline(CV, color="red", label = "UCL")
    plt.text(text_position[0],text_position[1],"UCL", fontsize=text_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = tick_labelsize)

    return fig_control_chart
 

