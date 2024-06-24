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
#  Auxiliary functions required by the self-simulation algorithm.

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, GRU
from tensorflow.keras.optimizers import Adam
import time
from tqdm import tqdm


def write_random_strains(strains_path, n_dim, number_strain_sequences, 
             sequence_lenght, max_abs_value_deformation, type_strains):
        def calculate_eq_strain(strains):
            trace = (np.sum(strains[:, 0:3], axis = 1)/3).reshape(strains.shape[0], 
                                                                  1)         
            dev_strains = np.copy(strains) #Initialisation of dev_strains array
            dev_strains[:, 0:3] = dev_strains[:, 0:3] - trace
            eq_strains = (((2/3)*np.sum((dev_strains*dev_strains), axis = 1))**0.5
                          )[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
            return eq_strains
        
        x_train = np.zeros([number_strain_sequences, sequence_lenght, 6])
        print("Calculating strains")
        time.sleep(0.3)
        mu, sigma = 0, 1 #mean and variance
            ##################################
        for i in tqdm(range(0, number_strain_sequences)):
            # if (type_strains == "control_points"):
            #     control_points = np.zeros([number_control_points, 6])
            #     if (n_dim == 2):
            #         control_points[:, 0:3] = np.random.normal(mu, sigma, \
            #                                       size = (number_control_points, 
            #                                               3)) #Normal
            #         # control_points[:, 0:3]  = control_points[:, 0:3] * 2 * \
            #         #                           max_abs_value_deformation - max_abs_value_deformation
            #         control_points[:, 3] = control_points[:, 2]
            #         control_points[:, 2] = 0
            #     else:
            #         control_points[:, :] = np.random.normal(mu, sigma, size = (number_control_points, 6)) #Normal
            #     total_accumulated_strain = np.cumsum(control_points, axis = 0)
            # else: #if (type_strains == "random"):
            strains = np.zeros([sequence_lenght, 6])
            if (n_dim == 2):
                strains[:, 0:3] = np.random.normal(mu, sigma, \
                                              size = (sequence_lenght, 
                                                      3)) #Normal
                # control_points[:, 0:3]  = control_points[:, 0:3] * 2 * \
                #                           max_abs_value_deformation - max_abs_value_deformation
                strains[:, 3] = strains[:, 2]
                strains[:, 2] = 0
            else:
                strains[:, :] = np.random.normal(mu, sigma, \
                                              size = (sequence_lenght, 
                                                      6)) 
            pass
            total_accumulated_strain = np.cumsum(strains, axis = 0)
            eq_strains = calculate_eq_strain(np.copy(total_accumulated_strain)) 
            max_eq_strain = np.max(abs(eq_strains), axis = 0)
            ##################################

            if (max_eq_strain > max_abs_value_deformation): #Remove 0.05 and use a parameter            
                if (type_strains == "control_points"):
                    control_points = control_points * (max_abs_value_deformation/max_eq_strain)  
                    # Check that it was limited properly
                    total_accumulated_strain = np.cumsum(control_points, axis = 0)
                else:
                    strains = strains * (max_abs_value_deformation/max_eq_strain)  
                    # Check that it was limited properly
                    total_accumulated_strain = np.cumsum(strains, axis = 0)
                    pass
                eq_strains = calculate_eq_strain(np.copy(total_accumulated_strain)) 
                max_eq_strain = np.max(abs(eq_strains), axis = 0)
                print("max_eq_strain is ", max_eq_strain)
            
            # if (type_strains == "control_points"):
            #     strains = np.zeros([sequence_lenght + 1, 6])
            #     # strains[1:, :] = interpolate_linearly(control_points, 
            #     #                                       number_interpolation_points)
            #     strains[1:, :] = interpolate_linearly(np.cumsum(control_points, axis = 0), 
            #                                           number_interpolation_points)
            # np.savetxt('./' + strains_path + '/' + '%i.txt' %(i), strains, 
            #                         delimiter = ' ')
            # np.savetxt('./' + strains_path + '/' + '%i.txt' %(i), 
            #            np.cumsum(strains, axis = 0), delimiter = ' ')
            x_train[i, :, :] = np.cumsum(strains, axis = 0)
            print("")
        print("Strains calculation finished")
        time.sleep(0.3)
        return (x_train)


def createNN(ML_MODEL_IS_3X3, numberNeuronHiddenLayers):
    if (ML_MODEL_IS_3X3):
        model = Sequential()
        model.add(Dense(units = numberNeuronHiddenLayers, 
                        kernel_initializer = 'he_normal', 
                        activation = 'relu', input_shape = (None, 3)))
        model.add(Dense(units = 3, kernel_initializer = 'he_normal', 
                        activation = 'linear'))
    else:
        model = Sequential()
        model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 'he_normal', 
                        activation = 'relu', input_shape = (None, 6)))
        model.add(Dense(units = 6, kernel_initializer = 'he_normal', 
                        activation = 'linear'))
    return model

def compileNN(model, slow):
    if (slow):
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.01,
            beta_1=0.99,
            beta_2=0.999999, 
            epsilon=1e-08, 
            amsgrad=True, 
        )
        model.compile(optimizer=opt, loss='mse')
    else:
        # model.compile(optimizer=Adam(lr = 0.001), loss='mse')
        model.compile(optimizer=Adam(lr = 0.001), loss='mse')

def createRNN(ML_Model_is_3x3, numberNeuronHiddenLayers):
    if (ML_Model_is_3x3):
        model = Sequential()
        # model.add(GRU(numberNeuronHiddenLayers,  dropout=0.25,  recurrent_dropout=0.25,
        model.add(GRU(numberNeuronHiddenLayers,
                      return_sequences = True,
                            input_shape = [None, 3])),
        model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
                              'he_normal', activation = 'relu'))
        model.add(Dense(units = 3, 
                             kernel_initializer = 'he_normal',
                             activation = 'linear'))   
    else:
        model = Sequential()
        # model.add(GRU(numberNeuronHiddenLayers,  dropout=0,  recurrent_dropout=0.5,
        #               return_sequences = True,
        #                     input_shape = [None, 6])),
        # model.add(GRU(numberNeuronHiddenLayers, return_sequences = True,
        #                     input_shape = [None, 6])),
        model.add(GRU(numberNeuronHiddenLayers, return_sequences = True,
                            input_shape = [None, 6])),        
        # model.add(LSTM(numberNeuronHiddenLayers, return_sequences = True,
        #                     input_shape = [None, 6])),
        # model.add(GRU(numberNeuronHiddenLayers, return_sequences = True)),
        # model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
        #                       'he_normal', activation = 'relu'))
        model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
                              'he_normal', activation = 'relu'))
        # model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
        #                       'he_normal', activation = 'relu'))       
        # model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
        #                      'he_normal', activation = 'relu'))       
        model.add(Dense(units = 6, 
                              kernel_initializer = 'he_normal',
                              activation = 'linear'))  
        # model.add(Dense(units = 3, 
        #                      kernel_initializer = 'he_normal',
        #                      activation = 'linear'))  
    return model