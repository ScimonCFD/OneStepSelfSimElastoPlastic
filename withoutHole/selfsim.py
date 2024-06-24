# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
#  This routine trains a neural network to replace a linear elastic 
#  Hookean law, following the procedure presented in the selfSimulation 
#  algorithm and using OpenFOAM + Python via pythonPal4foam. 

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import os
import numpy as np
import auxiliary_functions
from auxiliary_functions import *
from joblib import dump
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import random
import input_file
from input_file import *
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from distutils.dir_util import mkpath
import matplotlib.pyplot as plt
from functions import *
import time

# record start time
start = time.time()

# Set the default text font size
plt.rc('font', size=16)# Set the axes title font size
plt.rc('axes', titlesize=20)# Set the axes labels font size
plt.rc('axes', labelsize=20)# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)# Set the legend font size
plt.rc('legend', fontsize=18)# Set the font size of the figure title
plt.rc('figure', titlesize=24)

# Create a folder to plot the results
mkpath(ROUTE_NN_MODEL + "Results/")

# Create and train the initial linear regression
terminal("cd " + ROUTE_TO_NEURAL_NETWORK_CODE + " && python main.py")

# Run the theoretical simulation
terminal("cd " + ROUTE_THEORETICAL_MODEL + " && ./Allrun") 

# Create a mae loss function
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Suppress/hide the warning
np.seterr(invalid='ignore')

withNoise = True

n_mesh_train_set = 10

cont = 1

# Assemble the data set with the expected data
for load_inc in range(1, TOTAL_LOAD_INCREMENTS+1):
# for load_inc in range(1, TOTAL_LOAD_INCREMENTS):
    epsilon_Right = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) +
                                "/", "epsilon_Right")
    epsilon_Down = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) +
                               "/", "epsilon_Down")
    epsilon_Up = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) +
                             "/", "epsilon_Up")    
    sigma_Right = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) +
                              "/", "sigma_Right")
    sigma_Down = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                             "/", "sigma_Down")
    sigma_Up = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + "/",
                           "sigma_Up")  
    
    elastic_eps_Right = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                    str(int(load_inc)) + "/", "epsilon_Right")
    elastic_eps_Down = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                   str(int(load_inc)) + "/", "epsilon_Down")
    elastic_eps_Up = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                 str(int(load_inc)) + "/", "epsilon_Up")
    elastic_sig_Right = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                    str(int(load_inc)) + "/", "sigma_Right")
    elastic_sig_Down = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                   str(int(load_inc)) + "/", "sigma_Down")
    elastic_sig_Up = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                 str(int(load_inc)) + "/", "sigma_Up")
    
    if(load_inc == 1):
        orig_x_train = deserialise(ROUTE_THEORETICAL_MODEL + 
                                   str(int(load_inc)) + "/", "epsilon")
        
        sigma_expected = deserialise(ROUTE_THEORETICAL_MODEL + 
                                     str(int(load_inc)) +"/", "sigma")
        
        elastic_eps_set = deserialise(ROUTE_NN_MODEL + "fullyElastic/" +
                                      str(int(load_inc)) + "/", "epsilon")
        
        elastic_sig_set = deserialise(ROUTE_NN_MODEL + "fullyElastic/" + 
                                      str(int(load_inc)) + "/", "sigma")
        
    else:
        x_train_temp = deserialise(ROUTE_THEORETICAL_MODEL + str(int(load_inc)) 
                                   + "/", "epsilon")
        orig_x_train = np.concatenate((orig_x_train, x_train_temp), axis=0)
        
        sigma_temp = deserialise(ROUTE_THEORETICAL_MODEL + str(int(load_inc)) +
                                 "/", "sigma")
        sigma_expected = np.concatenate((sigma_expected, sigma_temp), axis=0)
        

        elastic_eps_temp = deserialise(ROUTE_NN_MODEL + "fullyElastic/" +
                                       str(int(load_inc)) + "/", "epsilon")
        elastic_eps_set = np.concatenate((elastic_eps_set, elastic_eps_temp), 
                                         axis=0)
        
        elastic_sig_temp = deserialise(ROUTE_NN_MODEL + "fullyElastic/" +
                                       str(int(load_inc)) + "/", "sigma")
        elastic_sig_set = np.concatenate((elastic_sig_set, elastic_sig_temp), 
                                         axis=0)        
        
    orig_x_train = np.concatenate((orig_x_train, epsilon_Right, epsilon_Down,
                                   epsilon_Up), axis = 0)
    
    sigma_expected = np.concatenate((sigma_expected, sigma_Right, sigma_Down,
                                     sigma_Up), axis = 0)
    
    elastic_eps_set = np.concatenate((elastic_eps_set, elastic_eps_Right, 
                                      elastic_eps_Down, elastic_eps_Up), 
                                     axis = 0)
    
    elastic_sig_set = np.concatenate((elastic_sig_set, elastic_sig_Right, 
                                      elastic_sig_Down, elastic_sig_Up), 
                                     axis = 0)
    
elastic_eps_set = elastic_eps_set.reshape([sequence_lenght, 
                                           int(elastic_eps_set.shape[0] / 
                                               sequence_lenght),
                                           int(elastic_eps_set.shape[1])])

elastic_sig_set = elastic_sig_set.reshape([sequence_lenght, 
                                           int(elastic_sig_set.shape[0] / 
                                               sequence_lenght),
                                           int(elastic_sig_set.shape[1])])

elastic_eps_set = np.swapaxes(elastic_eps_set, 0, 1)
elastic_sig_set = np.swapaxes(elastic_sig_set, 0, 1)

# Save the strains to a NumPy file
serialise(orig_x_train, ROUTE_TO_NEURAL_NETWORK_CODE, "orig_x_train")

# Save the strains to a NumPy file
serialise(sigma_expected, ROUTE_TO_NEURAL_NETWORK_CODE, "orig_y_train")

# The training set is a subset of the original strains
if (SUBSAMPLE_ORIGINAL_STRAINS):
    orig_x_train = deserialise(ROUTE_TO_NEURAL_NETWORK_CODE, "x_train")
    orig_x_train_scaled = deserialise(ROUTE_TO_NEURAL_NETWORK_CODE, 
                                      "x_train_scaled")

if (ML_MODEL_IS_3X3):
    dum_orig_x_train = np.zeros((int(orig_x_train.shape[0]), 
                                 int(orig_x_train.shape[1]), 6))
    dum_orig_x_train[:, :, :2] = orig_x_train[:, :, :2]
    dum_orig_x_train[:, :, 3] = orig_x_train[:, :, 2]    
    orig_x_train = np.copy(dum_orig_x_train)

    dum_orig_x_train_scaled = np.zeros((int(orig_x_train_scaled.shape[0]),
                                            int(orig_x_train_scaled.shape[1]),
                                            6))
    dum_orig_x_train_scaled[:, :, :2] = orig_x_train_scaled[:, :, :2]
    dum_orig_x_train_scaled[:, :, 3] = orig_x_train_scaled[:, :, 2]    
    orig_x_train_scaled = np.copy(dum_orig_x_train_scaled)

# Bring the neural network and the scalers to the parent folder
terminal("cp " + ROUTE_TO_NEURAL_NETWORK_CODE + "ML_model.h5 " + 
         ROUTE_NN_MODEL)

terminal("cp " + ROUTE_TO_NEURAL_NETWORK_CODE + "*scaler.joblib " + 
         ROUTE_NN_MODEL)

# Load the initial neural network
ML_model= keras.models.load_model(ROUTE_NN_MODEL +"ML_model.h5")

# Load the stresses used at training stage
orig_y_train = deserialise(ROUTE_TO_NEURAL_NETWORK_CODE, "y_train")   
orig_y_train_scaled = deserialise(ROUTE_TO_NEURAL_NETWORK_CODE, 
                                  "y_train_scaled")    
     
if (ML_MODEL_IS_3X3):
    dum_orig_y_train = np.zeros((int(orig_y_train.shape[0]), 
                                 int(orig_y_train.shape[1]), 6))
    dum_orig_y_train[:, :, :2] = orig_y_train[:, :, :2]
    dum_orig_y_train[:, :, 3] = orig_y_train[:, :, 2]    
    orig_y_train = np.copy(dum_orig_y_train) 

    dum_orig_y_train_scaled = np.zeros((int(orig_y_train_scaled.shape[0]),
                                        int(orig_y_train_scaled.shape[1]), 6))
    dum_orig_y_train_scaled[:, :, :2] = orig_y_train_scaled[:, :, :2]
    dum_orig_y_train_scaled[:, :, 3] = orig_y_train_scaled[:, :, 2]    
    orig_y_train_scaled = np.copy(dum_orig_y_train_scaled) 

if (withNoise):
    sigma_expected = sigma_expected.reshape([TOTAL_LOAD_INCREMENTS, 
                                         int(sigma_expected.shape[0] / 
                                             TOTAL_LOAD_INCREMENTS), 6])
    
else:
    sigma_expected = sigma_expected.reshape([TOTAL_LOAD_INCREMENTS,
                                             orig_y_train.shape[0], 6])

sigma_expected = np.swapaxes(sigma_expected, 0, 1)

# Load the scalers
x_scaler = load(ROUTE_NN_MODEL + 'x_scaler.joblib')
y_scaler = load(ROUTE_NN_MODEL + 'y_scaler.joblib')
    
mse_eps_int = []
mse_sig_int = []
mse_sig_int2 = []


for pass_num in range(TOTAL_NUMBER_PASSES):
    for i in range(TOTAL_ITERATIONS):     
        if (not ELASTIC_TRAINING):
            # Run the theoretical simulation
            terminal("cd " + ROUTE_THEORETICAL_MODEL + " && ./Allrun") 
    
        # Run load-driven simulation
        terminal("cd " + ROUTE_NN_MODEL + " && ./Allrun")

        if (pass_num == 0):
            if (cont == 1):
                master_x_train = np.zeros(
                                        [TOTAL_NUMBER_PASSES + 
                                         n_mesh_train_set, 
                                         int(orig_x_train.shape[0]/
                                             n_mesh_train_set), 
                                         TOTAL_LOAD_INCREMENTS, 6])
                
                master_y_train = np.zeros(
                                        [TOTAL_NUMBER_PASSES + 
                                         n_mesh_train_set,
                                        int(orig_y_train.shape[0]/
                                            n_mesh_train_set),
                                        TOTAL_LOAD_INCREMENTS, 6])
                
                master_x_train_scaled = np.copy(master_x_train)
                master_y_train_scaled = np.copy(master_y_train)

                master_x_train [:n_mesh_train_set, :,:,:] = \
                  np.copy(orig_x_train.reshape([n_mesh_train_set, 
                                                int(orig_x_train.shape[0]/
                                                    n_mesh_train_set),
                                                int(orig_x_train.shape[1]), 
                                                6]))
                
                master_y_train [:n_mesh_train_set, :,:,:] = np.copy(
                    orig_y_train.reshape([n_mesh_train_set, 
                                          int(orig_x_train.shape[0]/
                                              n_mesh_train_set),
                                          int(orig_x_train.shape[1]), 6])) 
                
 
                master_x_train_scaled[:n_mesh_train_set,:,:,:]= np.copy(
                    orig_x_train_scaled.reshape([n_mesh_train_set, 
                                                 int(orig_x_train.shape[0] / 
                                                     n_mesh_train_set), 
                                                 int(orig_x_train.shape[1]), 
                                                 int(orig_x_train.shape[2])]
                                                ))
                
                master_y_train_scaled[:n_mesh_train_set,:,:,:]= np.copy(
                    orig_y_train_scaled.reshape([n_mesh_train_set, 
                                                 int(orig_x_train.shape[0]/
                                                     n_mesh_train_set),
                                                 int(orig_x_train.shape[1]),
                                                 int(orig_x_train.shape[2])])) 
               
                x_temp = np.zeros([TOTAL_LOAD_INCREMENTS, 
                                   int(orig_x_train.shape[0] / 
                                       n_mesh_train_set), 
                                   orig_x_train.shape[2]])
                
                eps_temp_tot = np.zeros([TOTAL_LOAD_INCREMENTS,
                                      int(orig_x_train.shape[0] /
                                          n_mesh_train_set), 6])
                
                eps_temp_tot_orig_ML_model = np.zeros([TOTAL_LOAD_INCREMENTS,
                                                    int(orig_x_train.shape[0] /
                                                         n_mesh_train_set), 6])
                sig_temp_tot_orig_ML_model = np.zeros([TOTAL_LOAD_INCREMENTS,
                                                    int(orig_x_train.shape[0] /
                                                         n_mesh_train_set), 6])

        eps_calc_tot = np.zeros([TOTAL_LOAD_INCREMENTS, 
                                 int(orig_x_train.shape[0] / 
                                     n_mesh_train_set), 6])

        y_temp =  np.zeros([TOTAL_LOAD_INCREMENTS, int(orig_x_train.shape[0] /
                                                       n_mesh_train_set), 6])  
            
        for k in range(1, TOTAL_LOAD_INCREMENTS + 1):
            if (cont == 1):
                eps_temp = deserialise(ROUTE_THEORETICAL_MODEL + str(int(k)) +
                                       "/", "epsilon")
                eps_Right_temp = deserialise(ROUTE_THEORETICAL_MODEL + 
                                             str(int(k)) + "/", 
                                             "epsilon_Right")
                eps_Down_temp = deserialise(ROUTE_THEORETICAL_MODEL + 
                                            str(int(k)) + "/", "epsilon_Down")
                eps_Up_temp = deserialise(ROUTE_THEORETICAL_MODEL + 
                                          str(int(k)) + "/", "epsilon_Up")  
                
                eps_temp_orig_ml = deserialise(ROUTE_TO_RESULTS_ORIG_ML_MODEL +
                                               str(int(k)) + "/", "epsilon")
                eps_Right_temp_orig_ml = deserialise(
                                               ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                             str(int(k)) + "/", 
                                             "epsilon_Right")
                eps_Down_temp_orig_ml = deserialise(
                                               ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                            str(int(k)) + "/", "epsilon_Down")
                eps_Up_temp_orig_ml = deserialise(
                                               ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                          str(int(k)) + "/", "epsilon_Up")                  

                eps_temp_tot[k-1, :, :]= np.concatenate((eps_temp, 
                                                         eps_Right_temp, 
                                                         eps_Down_temp, 
                                                         eps_Up_temp), 
                                                        axis = 0)
                
                eps_temp_tot_orig_ML_model[k-1, :, :]= np.concatenate((
                                                        eps_temp_orig_ml, 
                                                        eps_Right_temp_orig_ml, 
                                                        eps_Down_temp_orig_ml, 
                                                        eps_Up_temp_orig_ml), 
                                                        axis = 0)
                
                
                sigma_temp_orig_ml = deserialise(ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                         str(int(k)) + "/", "sigma")
                sigma_Right_temp_orig_ml = deserialise(
                                                   ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                               str(int(k)) + "/", "sigma_Right")
                sigma_Down_temp_orig_ml = deserialise(
                                                   ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                              str(int(k)) + "/", "sigma_Down")
                sigma_Up_temp_orig_ml = deserialise(
                                                   ROUTE_TO_RESULTS_ORIG_ML_MODEL + 
                                            str(int(k)) + "/", "sigma_Up")
                
                sig_temp_tot_orig_ML_model[k-1, :, :] = np.concatenate(
                                                   (sigma_temp_orig_ml, 
                                                    sigma_Right_temp_orig_ml, 
                                                    sigma_Down_temp_orig_ml, 
                                                    sigma_Up_temp_orig_ml), 
                                                   axis = 0)

                
                
                
            eps_calc = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                   "epsilon")
            eps_calc_Right = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                         "epsilon_Right")
            eps_calc_Down = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                        "epsilon_Down")
            eps_calc_Up = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                      "epsilon_Up")   

            eps_calc_tot[k-1, :, :]= np.concatenate((eps_calc, eps_calc_Right,
                                                     eps_calc_Down, 
                                                     eps_calc_Up), axis = 0)
            
            sigma_temp = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                     "sigma")
            sigma_Right_temp = deserialise(ROUTE_NN_MODEL + str(int(k)) + "/", 
                                           "sigma_Right")
            sigma_Down_temp = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                          "sigma_Down")
            sigma_Up_temp = deserialise(ROUTE_NN_MODEL +  str(int(k)) + "/", 
                                        "sigma_Up")
            

            

            y_temp[k-1, :, :] = np.concatenate((sigma_temp, sigma_Right_temp, 
                                                sigma_Down_temp,  
                                                sigma_Up_temp), axis = 0)


        y_temp = np.swapaxes(y_temp, 0, 1)
        eps_calc_tot = np.swapaxes(eps_calc_tot, 0, 1)
    
        if (cont == 1):
            x_temp = np.copy(eps_temp_tot)
            x_temp = np.swapaxes(x_temp, 0, 1)
            x_temp[:, 1:, :] = np.diff(x_temp, axis = 1)
            eps_temp_tot = np.swapaxes(eps_temp_tot, 0, 1)
            eps_temp_tot_orig_ML_model = np.swapaxes(eps_temp_tot_orig_ML_model, 0, 1)
            sig_temp_tot_orig_ML_model = np.swapaxes(sig_temp_tot_orig_ML_model, 0, 1)
            
            
                
            if (ML_MODEL_IS_3X3):
                x_temp = np.delete(x_temp, [2, 4, 5], axis = 2)
        
            x_temp_scaled = x_scaler.transform(x_temp.reshape([
                                       x_temp.shape[0] * x_temp.shape[1], 
                                       x_temp.shape[2]])).reshape(x_temp.shape)

        if (ML_MODEL_IS_3X3):
            y_temp = np.delete(y_temp, [2, 4, 5], axis = 2)
            
        y_temp_scaled = y_scaler.transform(y_temp.reshape(
                                      [y_temp.shape[0]* y_temp.shape[1], 
                                       y_temp.shape[2]])).reshape(y_temp.shape)
            
        if (ML_MODEL_IS_3X3):
            ######### Y just for ref ###########
            y_for_ref = y_scaler.inverse_transform(
                ML_model.predict(x_temp_scaled).reshape(
                                [x_temp_scaled.shape[0]*x_temp_scaled.shape[1], 
                                x_temp_scaled.shape[2]]))
            mse_sig_int2.append(mse(y_for_ref.reshape(x_temp_scaled.shape), 
                                    np.delete(sigma_expected, [2, 4, 5], 
                                              axis = 2)).numpy()/1e10)

        else:
                      
            ######### Y just for ref ###########
            y_for_ref = y_scaler.inverse_transform(
                ML_model.predict(x_temp_scaled).reshape(
                                [x_temp_scaled.shape[0]*x_temp_scaled.shape[1], 
                                 x_temp_scaled.shape[2]]))
            mse_sig_int2.append(mse(y_for_ref.reshape(x_temp_scaled.shape), 
                                    sigma_expected).numpy()/1e10)
        
        if (ML_MODEL_IS_3X3):
            master_x_train_scaled[pass_num+n_mesh_train_set,:,:,0] = np.copy(
                x_temp_scaled[:, :, 0])
            master_x_train_scaled[pass_num+n_mesh_train_set,:,:,1] = np.copy(
                x_temp_scaled[:, :, 1])
            master_x_train_scaled[pass_num+n_mesh_train_set,:,:,3] = np.copy(
                x_temp_scaled[:, :, 2])

            master_y_train_scaled[pass_num+n_mesh_train_set,:,:,0] = np.copy(
                y_temp_scaled[:, :, 0])
            master_y_train_scaled[pass_num+n_mesh_train_set,:,:,1] = np.copy(
                y_temp_scaled[:, :, 1])
            master_y_train_scaled[pass_num+n_mesh_train_set,:,:,3] = np.copy(
                y_temp_scaled[:, :, 2])
                                                    
                          
            master_x_train[pass_num+n_mesh_train_set, :, :, 0] = np.copy(
                x_temp[:, :, 0])  
            master_x_train[pass_num+n_mesh_train_set, :, :, 1] = np.copy(
                x_temp[:, :, 1])  
            master_x_train[pass_num+n_mesh_train_set, :, :, 3] = np.copy(
                x_temp[:, :, 2])  
            
            master_y_train[pass_num+n_mesh_train_set, :, :, 0] = np.copy(
                y_temp[:, :, 0])  
            master_y_train[pass_num+n_mesh_train_set, :, :, 1] = np.copy(
                y_temp[:, :, 1])  
            master_y_train[pass_num+n_mesh_train_set, :, :, 3] = np.copy(
                y_temp[:, :, 2])  
            
        else:
            master_x_train_scaled[pass_num+n_mesh_train_set,:,:,:] = \
                np.copy(x_temp_scaled)    
            master_y_train_scaled[pass_num+n_mesh_train_set,:,:,:] = \
                np.copy(y_temp_scaled)
            master_x_train[pass_num+n_mesh_train_set, :, :, :] = \
                x_temp  
            master_y_train[pass_num+n_mesh_train_set, :, :, :] = \
                y_temp

        if (WITH_MOVING_WINDOW and (pass_num >= SETS_IN_MOVING_WINDOW)):
            shape1 = master_x_train_scaled[:n_mesh_train_set, :,:,:].shape
            shape2 = master_x_train_scaled[
                -1 * (SETS_IN_MOVING_WINDOW):, :,:,:].shape
            x_train = np.concatenate((
                master_x_train_scaled[:n_mesh_train_set, :,:,:].reshape(
                    [shape1[0]*shape1[1], shape1[2], shape1[3]]), 
                master_x_train_scaled[(
                   (pass_num) + (n_mesh_train_set+1) - SETS_IN_MOVING_WINDOW):(
                       (pass_num) + (n_mesh_train_set+1)), :,:,:].reshape(
                       [shape2[0] * shape2[1], shape2[2], shape2[3]]), ), 
                axis = 0)
            y_train = np.concatenate((
                master_y_train_scaled[:n_mesh_train_set, :,:,:].reshape([
                    shape1[0]*shape1[1], shape1[2], shape1[3]]), 
                master_y_train_scaled[(
                   (pass_num) + (n_mesh_train_set+1) - SETS_IN_MOVING_WINDOW):(
                        (pass_num) + (n_mesh_train_set+1)), :,:,:].reshape([
                            shape2[0]*shape2[1], shape2[2], shape2[3]]), ),
                    axis = 0)
            
            if (ML_MODEL_IS_3X3):
                x_train = np.delete(x_train, [2, 4, 5], axis = 2)
                y_train = np.delete(y_train, [2, 4, 5], axis = 2)    
                
        else:
 
            if (ML_MODEL_IS_3X3):
                x_train = np.delete(np.concatenate(
                  master_x_train_scaled[:pass_num+n_mesh_train_set+1, :, :, :], 
                  axis = 0), [2, 4, 5], axis = 2)
                y_train = np.delete(np.concatenate(
                  master_y_train_scaled[:pass_num+n_mesh_train_set+1, :, :, :], 
                  axis = 0), [2, 4, 5], axis = 2)    
     
            else:
                x_train = np.concatenate(
                  master_x_train_scaled[:pass_num+n_mesh_train_set+1, :, :, :], 
                  axis = 0)
                y_train = np.concatenate(
                  master_y_train_scaled[:pass_num+n_mesh_train_set+1, :, :, :], 
                  axis = 0)    

        history = ML_model.fit(x_train, y_train, epochs = 5000) 
        ML_model.save(ROUTE_NN_MODEL + "ML_model.h5")
        
        if (ML_MODEL_IS_3X3):
            mse_eps_int.append(mse(eps_calc_tot, eps_temp_tot).numpy())
            mse_sig_int.append(mse(y_temp, np.delete(sigma_expected, [2, 4, 5],
                                                     axis = 2)).numpy())
        else:
            mse_eps_int.append(mse(eps_calc_tot, eps_temp_tot).numpy())
            mse_sig_int.append(mse(y_temp, sigma_expected).numpy())
        
        mkpath(ROUTE_NN_MODEL + "Results/plots/" + str(cont) + "/" )
        
        fig = plt.figure(figsize=(15, 10))
        plt.plot(range(len(mse_eps_int)),  mse_eps_int,  color = "blue")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        fig.suptitle("MSE Epsilon over time steps", fontsize=20)
        # plt.legend()
        # plt.show()
        # fig.savefig(ROUTE_NN_MODEL + "Results/plots/" + str(pass_num)+ 
        # "/MSE_Epsilon.png")
        fig.savefig(ROUTE_NN_MODEL + "Results/plots/" + "/MSE_Epsilon.png", 
                    bbox_inches='tight')
        plt.close(fig)
        
        fig = plt.figure(figsize=(15, 10))
        plt.plot(range(len(mse_sig_int)),  mse_sig_int,  color = "blue")
        # plt.plot(range(6),  MSE_sigma[:6],  color = "blue")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        fig.suptitle("MSE Sigma over time steps", fontsize=20)
        # plt.legend()
        # plt.show()
        # fig.savefig(ROUTE_NN_MODEL + "Results/plots/" + str(pass_num)+  
        # "/MSE_Sigma.png")
        fig.savefig(ROUTE_NN_MODEL + "Results/plots/" +  "/MSE_Sigma.png", 
                    bbox_inches='tight')
        plt.close(fig)


        if (ML_MODEL_IS_3X3):
            component = ["xx", "xy", "yy"]
            dum_ind = (0, 1, 2)
        else:
            component = ["xx", "xy", "xz", "yy", "yz", "zz"]
            dum_ind = (0, 1, 3)
        for com in range(len(component)):            
            # if (com in (0, 1, 3)):
            if (com in dum_ind):
                fig = plt.figure(figsize=(15, 10))
                if (ML_MODEL_IS_3X3):
                    plt.plot(np.delete(eps_temp_tot.swapaxes(0, 1), [2, 4, 5], 
                                       axis = 2)[-1, :, com], 
                                      np.delete(eps_temp_tot.swapaxes(0, 1), 
                                                [2, 4, 5], 
                                                axis = 2)[-1, :, com],
                                      color = "blue", label = "Expected", 
                                      marker='x')                
                    plt.scatter(np.delete(eps_temp_tot.swapaxes(0, 1), 
                                          [2, 4, 5], axis = 2)[-1, :, com],
                                np.delete(elastic_eps_set.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                color = "green", label = "Elastic", 
                                marker='*')                
                    plt.scatter(np.delete(eps_temp_tot.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                np.delete(eps_calc_tot.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                color = "red", label = "Calculated", 
                                marker='o', alpha=(0.3)) 
                    plt.scatter(np.delete(eps_temp_tot.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                np.delete(eps_temp_tot_orig_ML_model.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                color = "purple", label = "OriginalMLModel", 
                                marker='+', alpha=(0.3)) 
                else:
                    plt.plot(eps_temp_tot.swapaxes(0, 1)[-1, :, com], 
                             eps_temp_tot.swapaxes(0, 1)[-1, :, com], 
                             color = "blue", label = "Expected", marker='x')                
                    plt.scatter(eps_temp_tot.swapaxes(0, 1)[-1, :, com],
                                elastic_eps_set.swapaxes(0, 1)[-1, :, com], 
                                color = "green", label = "Elastic", marker='*')                
                    plt.scatter(eps_temp_tot.swapaxes(0, 1)[-1, :, com], 
                                eps_calc_tot.swapaxes(0, 1)[-1, :, com], 
                                color = "red", label = "Calculated",marker='o', 
                                alpha=(0.3))                
                plt.xlabel(r'Expected strain $(m/m)$')
                plt.ylabel(r'Calculated strain $(m/m)$')
                # fig.suptitle("Epsilon_" + component[com], fontsize=20)
                # plt.legend()
                fig.savefig(ROUTE_NN_MODEL + "Results/plots/" + str(cont) + 
                            "/epsilon_" + component[com] + ".png", 
                            bbox_inches='tight')
                plt.close(fig)

        for com in range(len(component)):
            if (com in dum_ind):
                fig = plt.figure(figsize=(15, 10))
                if (ML_MODEL_IS_3X3):
                    plt.plot(np.delete(sigma_expected.swapaxes(0, 1),[2, 4, 5],
                                       axis = 2)[-1, :, com],
                             np.delete(sigma_expected.swapaxes(0, 1),[2, 4, 5],
                                       axis = 2)[-1, :, com],
                             color = "blue", label = "Expected", marker='x')                
                    plt.scatter(np.delete(sigma_expected.swapaxes(0, 1), 
                                          [2, 4, 5], axis = 2)[-1, :, com],
                                np.delete(elastic_sig_set.swapaxes(0, 1), 
                                          [2, 4, 5], axis = 2)[-1, :, com],
                                color = "green", label = "Elastic", marker='*')                
                    plt.scatter(np.delete(sigma_expected.swapaxes(0, 1), 
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                y_temp.swapaxes(0, 1)[-1, :, com],
                                color = "red", label = "Calculated", 
                                marker='o', alpha=(0.3))  
                    plt.scatter(np.delete(sigma_expected.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                np.delete(sig_temp_tot_orig_ML_model.swapaxes(0, 1),
                                          [2, 4, 5], axis = 2)[-1, :, com], 
                                color = "purple", label = "OriginalMLModel", 
                                marker='+', alpha=(0.3)) 
                                    
                else:
                    plt.plot(sigma_expected.swapaxes(0, 1)[-1, :, com], 
                             sigma_expected.swapaxes(0, 1)[-1, :, com], 
                             color = "blue", label = "Expected", marker='x')                
                    plt.scatter(sigma_expected.swapaxes(0, 1)[-1, :, com], 
                                elastic_sig_set.swapaxes(0, 1)[-1, :, com], 
                                color = "green", label = "Elastic", marker='*')                
                    plt.scatter(sigma_expected.swapaxes(0, 1)[-1, :, com], 
                                y_temp.swapaxes(0, 1)[-1, :, com], 
                                color = "red", label = "Calculated", 
                                marker='o', alpha=(0.3))  
                    plt.scatter(sigma_expected.swapaxes(0, 1)[-1, :, com], 
                                y_temp.swapaxes(0, 1)[-1, :, com], 
                                color = "red", label = "Calculated", 
                                marker='o', alpha=(0.3))  
                plt.xlabel(r'Expected stress $(Pa)$')
                plt.ylabel(r'Calculated stress $(Pa)$')
                # fig.suptitle("Sigma_" + component[com], fontsize=20)
                # plt.legend()
                fig.savefig(ROUTE_NN_MODEL + "Results/plots/" + str(cont) + 
                            "/sigma" + component[com] + ".png", 
                            bbox_inches='tight')
                plt.close(fig)
        
        pd.DataFrame(history.history).plot(figsize=(15, 10))
        plot_name = (ROUTE_NN_MODEL + "Results/plots/" + str(cont) + 
        '/Convergence history.png')
        #Save a copy   of the ML model
        ML_model.save(ROUTE_NN_MODEL + "Results/plots/" + str(cont) + 
                      "/ML_model.h5")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close(fig)
        
        terminal("cd " + ROUTE_NN_MODEL + " && ./Allclean")        
        cont = cont + 1
        
end = time.time()
print("Calculation is finished.")

with open('./report.txt', 'a') as f:
    f.write("Total calculation time: " + str(end-start) + " (s) \n or \n " + 
            str((end-start)/3600) + "(h)")
    f.close()