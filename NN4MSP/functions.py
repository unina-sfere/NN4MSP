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
    
    """ Return the simulated data set, as described in the reference paper, to 
        train and design the Neural Network, and an array of zeros and ones, 
        corresponding to the negative and positive samples, respectivetly.

    # Parameters
        s: int, number of streams.
        k: int, subgroup size.
        num_pos_samples: int, number of positive samples. 
        num_neg_samples: int, number of negative samples.
        loc_res: float, mean of the distribution.
        scale_res: float, standard deviation of the distribution.
        set_seed: int, random number generator.

    # Returns
        X: ndarray of shape (num_pos_samples + num_neg_samples, s + 2).   
        y: ndarray of shape (num_pos_samples + num_neg_samples,).
    
    """
    
    np.random.seed(set_seed) 
    negative_samples = np.random.normal(loc = loc_res, scale = scale_res,  
                                    size = (num_neg_samples * k,s))   
    negative_samples = negative_samples.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() 
    overall_mean = negative_samples.mean(axis=1) 
    sample_range = negative_samples.max(axis=1) - negative_samples.min(axis=1) 
    negative_samples = np.c_[negative_samples,overall_mean,sample_range] 
    
    shift = [1,2,3] 
    positive_samples_all = np.zeros((1, s+2))
    
    for i in shift:
        for j in range(s-1): 
            for l in combinations([i for i in range(s)], j + 1):
                np.random.seed(0) # seed 
                positive_samples = np.random.normal(loc = loc_res, scale = scale_res, size = (num_pos_samples * k,s))
                positive_samples[:, np.array(l)] = positive_samples[:, np.array(l)] + i
                positive_samples = positive_samples.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() 
                overall_mean = positive_samples.mean(axis=1) 
                sample_range = positive_samples.max(axis=1) - positive_samples.min(axis=1) 
                positive_samples = np.c_[positive_samples,overall_mean,sample_range]
                positive_samples_all = np.vstack([positive_samples_all, positive_samples])
    positive_samples_all = np.delete(positive_samples_all, (0), axis=0) 
    
    X = np.vstack([negative_samples, positive_samples_all])
    y = np.repeat([0,1], X.shape[0]/2)
    
    return X,y
    

def ROC_AUC_plot(classifier, X_val, y_val, figure, xlabel = '', ylabel = '', legend_fontsize = 10, label_fontsize = 12, 
                text_fontsize = 10, text_position = [0.2,0.7], tick_labelsize = 10):
    
    """ Plot the Receiver operating characteristic curve (ROC) and compute the
        Area Under the ROC from prediction scores.

    # Parameters
        classifier: tensorflow.python.keras.engine.sequential.Sequential object,
            trained Multilayer Perceptron classifier model.
        X_val: array, data on which to compute ROC and AUC. 
        y_val: array, true labels or binary label indicators.   
        figure: a Figure instance.
        xlabel: str, a title for the x axis.
        ylabel: str, a title for the y axis.
        legend_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, 
            font size of the legend, the default value is 10. 
        label_fontsize: int, label font size for both axes, the default value is 12.
        text_fontsize: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, 
            font size of the text, the default value is 10.       
        text_position: list, the position to place the text, the default value is [0.2,0.7].  
        tick_labelsize: float or str, tick label font size, the default value is 10.  
    
    # Returns
        figure: a matplotlib.figure.Figure object.
        
    """    

     
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

    return figure

def set_cv_alpha(n, s, k, loc_res, scale_res ,set_seed, scaler, classifier, cv):

    """ Return the Type-I error corrisponding to the given cut-off value of the 
        neuron in the output layer.

    # Parameters
        n: int, number of the drawn samples from the parameterized normal distribution.
        s: int, number of streams.
        k: int, subgroup size.
        loc_res: float, mean of the distribution.
        scale_res: float, standard deviation of the distribution.
        num_pos_samples: int, number of positive samples. 
        num_neg_samples: int, number of negative samples.
        set_seed: int, random number generator.
        scaler: sklearn.preprocessing._data.StandardScaler object, standardize 
            features by removing the mean and scaling to unit variance.
        classifier: tensorflow.python.keras.engine.sequential.Sequential object,
            trained Multilayer Perceptron classifier model.
        cv: float, cut-off value of the neuron in the output layer. 

    # Returnso doc
        alpha: float.
    
    """    

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
    """ Return an istance of the Multilayer Perceptron (MLP) classifier.

    # Parameters
        hidden_activation_function: list, activation functions for the hidden layers.
        num_hidden_layer: int, number of hidden layers in the model.
        num_hidden_neuron: list, number of neurons in the the hidden layers

    # Returns
        classifier: A MLP model.
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

    """ Plot the control chart based on the Neural Network predicted probability.

    # Parameters
        NN_pred: 1d array-like, predicted probability, as returned by the classifier.
        fig_control_chart: a Figure instance.
        CV: float, the cut-off value of the neuron in the output layer
        xlabel: str, a title for the x axis.
        ylabel: str, a title for the y axis.
        text_position: list, the position to place the text, the default value is [35,0.9].
        text_fontsize: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, 
            font size of the text, the default value is 12.
        label_fontsize: int, label font size for both axes, the default value is 12    
        tick_labelsize: float or str, tick label font size, the default value is 10.         

    # Returns
        fig_control_chart: a matplotlib.figure.Figure object.
        
    """        
    
    
    x = np.arange(1,len(NN_pred) + 1,1)

    plt.plot(x, NN_pred, color='black', ls='-', marker='*')
    plt.axhline(CV, color="red", label = "UCL")
    plt.text(text_position[0],text_position[1],"UCL", fontsize=text_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = tick_labelsize)
    plt.xlim([0,len(NN_pred)])
    
    return fig_control_chart
 

