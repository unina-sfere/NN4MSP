NN4MSP 
================

This repository contains Python code and data of the paper of Lepore, Palumbo and Sposito 
*Neural network based control charting for multiple stream processes with an application to 
HVAC systems in passenger railway vehicles*. Note that this work has been done in the framework 
of the R&D project of the multiregional investment programme REINForce:REsearch to INspire the Future 
(CDS000609) with Hitachi Rail STS (https://www.hitachirail.com/), supported by the Italian Ministry 
for Economic Development (MISE) through the Invitalia agency.

This repository contains the following files:

-   NN4MSP/data contains the HVAC data set
-   NN4MSP/dataset.py allows the user to access the HVAC data set from the `NN4MSP`package
-   NN4MSP/functions.py is the source code of the Python package `NN4MSP` 
-   NN4MSP_tutorial.ipynb is the Jupyter Notebook performing all the analysis shown in
    the Section "*A real-case study*" of the paper

Moreover, in the following Section we provide a tutorial to show how to implement in Python 
the proposed methodology used in the paper to the real-case study.

# Neural network based control charting for multiple stream processes with an application to HVAC systems in passenger railway vehicles

## Introduction

This tutorial shows how to implement in Python the proposed methodology to the 
real-case study to monitor the HVAC systems installed on board of passenger railway vehicles. 
The operational data were acquired and made available by the rail transport company Hitachi 
Rail STS based in Italy.
HVAC data set contains the data analyzed in the paper and can be loaded by using the function `load_HVAC_data()`. 
Alternatively, one can use another data set and apply this methodology to any multiple stream process.

You can install the development version of the Python package `NN4MSP` from GitHub with

``` python
pip install git+https://github.com/unina-sfere/NN4MSP#egg=NN4MSP
```

You can install the Python package `NN4MSP` using pip

``` python
pip install NN4MSP
```

``` python

# Import libraries

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

from keras import Sequential
from keras.layers import Dense

from NN4MSP.functions import *
import NN4MSP.dataset

```
## Neural Network training

Set the simulation parameters to properly generate the data set to train the Neaural Network (NN).

``` python

# Simulation parameters

s = 6 # number of streams
k = 5 # subgroup size
num_neg_samples = 55800 # number of negative samples of k observations
num_pos_samples = 300 # number of positive samples of k observations for each OC scenario

loc_res = 0 # Mean of the distribution of the residuals
scale_res = 1 # Standard deviation of the distribution of the residuals

```
Then, call the function `dataset_generator` from the `MSPforNN` package to generate the data set, simulated according to the procedure 
described in the simulation section of the paper, and the corresponding vector of classes 0 (negative sample) and 1 (positive sample)

``` python

X, y = dataset_generator(s = s, k = k, num_neg_samples = num_neg_samples, num_pos_samples = num_pos_samples, loc_res = loc_res, 
                scale_res = scale_res, set_seed = 0)

```
Split the simulated data set into 70% training set and 30% validation set. 

``` python

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify = y ,random_state=27)

```
Standardize the features by removing the mean and scaling to unit variance.

``` python

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

```

Set the NN hyperparameters and train the NN with the function `NN_model` from the `MSPforNN` package. 

``` python

# NN hyperparameters

num_hidden_layer = 1 # Number of hidden layers
hidden_activation_function = ['relu'] # activation function in the hidden layer
number_hidden_neuron = [5] # number of neurons in the hidden layer

epochs = 10 # Number of epochs to train the model. An epoch is an iteration over the entire data set provided
batch_size = 256 # Number of samples per gradient update

# NN Training 

classifier = NN_model(hidden_activation_function = hidden_activation_function,
                   num_hidden_layer = num_hidden_layer, num_hidden_neuron = number_hidden_neuron) 

# Compiling the neural network

classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy']) # Configures the model for training

# Fitting 

history = classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_val, y_val)) # Trains the model

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/NN_fitting_history.PNG)

``` python

# History of training and validation accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

```

![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/model_accuracy.png)

Plot the Receiver Operating Characteristic (ROC) curve and compute the Area Under the Curve (AUC)
as performance measure to manually tune the typical NN hyperparameters. 
Use the function `ROC_AUC_plot` from the `MSPforNN` package.

``` python

fig_size = (5, 5)
f = plt.figure(figsize=fig_size)
f = ROC_AUC_plot(classifier, X_val, y_val, f, xlabel = 'False Positive Rate', ylabel = 'True Positive Rate')

```

![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/ROC_AUC_plot.png)

## Set the cut-off value of the neuron in the output layer

To allow fair comparison with the traditional statistical control charting procedures, the cut-off value (CV) of the neuron in the output layer
must be set and can be regarded as the key threshold to set the Type-I and Type-II errors.
A table of the CVs of the proposed NN corresponding to typical false alarm rate values is provided in the paper. Additionally, using the function `set_cv_alpha` from the `MSPforNN` package, you can compute the Type-I error corresponding to any CV. 

``` python

set_seed = 0
cv = 0.940 # cut-off value
n = 100000 # number of samples of 5 observations

alpha = set_cv_alpha(n = n, s = s, k = k, loc_res = loc_res, scale_res = scale_res , scaler = scaler, classifier = classifier, cv = cv, set_seed = set_seed)
print(alpha)

```

0.0027 \\
which is the default value for a Shewhart control chart in the 6-sigma quality approach.

## Real-case study: HVAC systems in passenger railway vehicles 

Import the HVAC datavset. Data have already been cleaned to remove unsteady working conditions and sensor
measurement errors and validated by domain expert.

``` python

HVAC_data = NN4MSP.dataset.load_HVAC_data()

```
### Phase I 

Filter `Vehicle` by Train_1 and consider 10 days of operational data from "07-27" to "08-08" for mean and variance estimations.

``` python

train_1_data = HVAC_data[HVAC_data["Vehicle"] == "Train_1"]
train_1_data = train_1_data.loc[(train_1_data['Timestamp'] >= '07-27')
                     & (train_1_data['Timestamp'] < '08-08')]
                     
```

Select only the DeltaTemp variables and compute the mean every 5 rows.   

``` python

train_1_data = train_1_data.iloc[:,-6:]
train_1_data = train_1_data.to_numpy() # Convert pandas dataframe to NumPy array
train_1_data_mean = train_1_data.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() 

```

Plot the average value of the DeltaTemp signals of all of the coaches of train 1 over 50 subgroup (10 minutes worth of data).

``` python

fig = plt.figure(figsize=(12, 6))

x = np.arange(1,51,1)

plt.plot(x,train_1_data_mean[210:260,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_1_data_mean[210:260,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_1_data_mean[210:260,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_1_data_mean[210:260,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_1_data_mean[210:260,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_1_data_mean[210:260,5], label = 'Coach 6', color='violet', ls='-', marker='P')
plt.xlabel('Subgroup', fontsize=12)
plt.ylabel('$ \Delta$T', fontsize=12)
plt.legend(fontsize=10)

plt.xlim([0,51])
plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)

plt.show()

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/plot_DeltaT_PhaseI_train_1.png)

Compute the residuals from each train coach and then calculate the mean and the variance.

``` python

train_1_residual = train_1_data_mean - np.mean(train_1_data_mean, axis = 1, keepdims= True) 

mean_res = np.mean(train_1_residual)
std_res = np.std(train_1_residual)

```

### Phase II

#### Train 2 

The following figure shows a MSP data in which we clearly see that the process is out of control 
and that an assignable cause affects the output from one stream

``` python

train_2_data = HVAC_data[HVAC_data["Vehicle"] == "Train_2"] # Filter Vehicle by Train 2 
train_2_data = train_2_data.iloc[0:-4,-6:] # Select the DeltaTemp variables 
train_2_data = train_2_data.to_numpy()
train_2_data_mean = train_2_data.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() # Average every 5 rows 

# Plot the ΔT signals from the six train coaches 

fig = plt.figure(figsize=(12, 6))

x = np.arange(1,31,1)

plt.plot(x,train_2_data_mean[235:265,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_2_data_mean[235:265,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_2_data_mean[235:265,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_2_data_mean[235:265,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_2_data_mean[235:265,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_2_data_mean[235:265,5], label = 'Coach 6', color='violet', ls='-', marker='P')
plt.xlabel('Subgroup', fontsize=12)
plt.ylabel('$ \Delta$T', fontsize=12)
plt.legend(fontsize=10)

plt.xlim([0,31])
plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)

plt.show()

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/plot_DeltaT_PhaseII_train_2.png)

After computing and standardizing the residuals from each coach, the range of the subgroup means of
the residuals and the overall mean at each sample time are calculated. 

``` python

# Definton of the input vector

train_2_residual = train_2_data_mean - np.mean(train_2_data_mean, axis = 1, keepdims= True)
train_2_mean_std = (train_2_residual - mean_res)/std_res
overall_mean = train_2_mean_std.mean(axis=1) 
sample_range = train_2_mean_std.max(axis=1) - train_2_mean_std.min(axis=1) 
train_2_mean_std = np.c_[train_2_mean_std,overall_mean,sample_range]

```
Then the input vector is given as input to the NN.

``` python

train_2_mean_std = scaler.transform(train_2_mean_std)
train_2_mean_std_pred = classifier.predict(train_2_mean_std)

```

Finally, you can plot the control chart based on the NN predicted probability by calling the function 
`control_chart` from the `MSPforNN` package.

``` python

fig_size = (12, 6)
fig_control_chart = plt.figure(figsize=fig_size)
fig_control_chart = control_chart(NN_pred = train_2_mean_std_pred[235:265], fig_control_chart = fig_control_chart, 
                                  CV = cv, xlabel = "Subgroup", ylabel = "Probability")

```

![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/PhaseII_train_2_controlchart.png)

The time-series plot of the residuals for each coach is displayed to help the practioner to identify how many and which stream(s) have shifted.

``` python

fig = plt.figure(figsize=(12, 6))

x = np.arange(1,31,1)

plt.plot(x,train_2_mean_std[235:265,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_2_mean_std[235:265,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_2_mean_std[235:265,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_2_mean_std[235:265,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_2_mean_std[235:265,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_2_mean_std[235:265,5], label = 'Coach 6', color='violet', ls='-', marker='P')
plt.xlabel('Subgroup', fontsize=12)
plt.ylabel('$ X_{tj} $', fontsize=12)
plt.legend(fontsize=10)

plt.xlim([0,31])
plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)

plt.show()

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/plot_residuals_PhaseII_train_2.png)

We can see that the coach 5 residuals are significantly higher than the other residuals, thus the HVAC system installed on-board coach 5 perform badly
and is not able to meet the required European regulations and ensure passenger thermal comfort.

#### Train 3 

The following figure shows a MSP data in which we clearly see that the process is out of control 
and that an assignable cause affects the output from four streams

``` python

train_3_data = HVAC_data[HVAC_data["Vehicle"] == "Train_3"]

train_3_data = train_3_data.loc[(train_3_data['Timestamp'] >= '07-25')
                     & (train_3_data['Timestamp'] < '07-26')]

train_3_data = train_3_data.iloc[0:-3,-6:]

train_3_data = train_3_data.to_numpy()
train_3_data_mean = train_3_data.transpose().reshape(-1,k).mean(1).reshape(s,-1).transpose() 

# Plot the ΔT signals from the six train coaches

fig = plt.figure(figsize=(12, 6))

x = np.arange(1,41,1)

plt.plot(x,train_3_data_mean[15:55,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_3_data_mean[15:55,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_3_data_mean[15:55,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_3_data_mean[15:55,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_3_data_mean[15:55,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_3_data_mean[15:55,5], label = 'Coach 6', color='violet', ls='-', marker='P')
plt.xlabel('Subgroup', fontsize=12)
plt.ylabel('$ \Delta$T', fontsize=12)
plt.legend(fontsize=10)

plt.xlim([0,41])
plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)

plt.show()

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/plot_DeltaT_PhaseII_train_3.png)

After computing and standardizing the residuals from each coach, the
range of the subgroup means of the residuals and the overall mean at each sample time are calculated. 

``` python

train_3_data_mean = train_3_data_mean - np.mean(train_3_data_mean, axis = 1, keepdims= True)
train_3_mean_std = (train_3_data_mean - mean_res)/std_res

overall_mean = train_3_mean_std.mean(axis=1) 
sample_range = train_3_mean_std.max(axis=1) - train_3_mean_std.min(axis=1) 
train_3_mean_std = np.c_[train_3_mean_std,overall_mean,sample_range]

```
Then the range, the overall mean and the six residuals for each coach are given as input to the NN.

``` python

train_3_mean_std = scaler.transform(train_3_mean_std)
train_3_mean_std_pred = classifier.predict(train_3_mean_std)

```

Finally, you can plot the control chart by calling the function `control_chart` from the `MSPforNN` package

``` python

fig_size = (12, 6)
fig_control_chart = plt.figure(figsize=fig_size)
fig_control_chart = control_chart(NN_pred = train_3_mean_std_pred[15:55], fig_control_chart = fig_control_chart, 
                                  CV = cv, xlabel = "Subgroup", ylabel = "Probability")

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/PhaseII_train_3_controlchart.png)

We can plot the residuals from each coach of the train 3.

``` python

fig = plt.figure(figsize=(12, 6))

x = np.arange(1,41,1)

plt.plot(x,train_3_mean_std[15:55,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_3_mean_std[15:55,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_3_mean_std[15:55,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_3_mean_std[15:55,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_3_mean_std[15:55,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_3_mean_std[15:55,5], label = 'Coach 6', color='violet', ls='-', marker='P')
plt.xlabel('Subgroup', fontsize=12)
plt.ylabel('$ X_{tj} $', fontsize=12)
plt.legend(fontsize=10)

plt.xlim([0,41])
plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)

plt.show()

```
![](https://github.com/unina-sfere/NN4MSP/blob/main/README_Figure/plot_residuals_PhaseII_train_3.png)

The above plot shows that coaches 1,2,4,5 of the train 3 perform differently from the other two coaches and helps 
the practitioner to obtain a correct interpretation of the OC situation.


