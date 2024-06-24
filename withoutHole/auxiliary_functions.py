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

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def createNN(ML_MODEL_IS_3X3, numberNeuronHiddenLayers):
    if (ML_MODEL_IS_3X3):
        model = Sequential()
        model = Sequential()
        model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
                        'he_normal', activation = 'relu', input_shape = 
                        (None, 3)))
        model.add(Dense(units = 3, kernel_initializer = 'he_normal', 
                        activation = 'linear'))
    else:
        model = Sequential()
        model.add(Dense(units = numberNeuronHiddenLayers, kernel_initializer = 
                        'he_normal', activation = 'relu', input_shape = 
                        (None, 6)))
        model.add(Dense(units = 6, kernel_initializer = 'he_normal', 
                        activation = 'linear'))
    return model

def compileNN(model, slow):
    if (slow):
        opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.99, 
                                       beta_2=0.999999, epsilon=1e-08, 
                                       amsgrad=True)
        model.compile(optimizer=opt, loss='mse')
    else:
        model.compile(optimizer=Adam(lr = 0.01), loss='mse')


def serialise(objectToSerialise, routeResultingFile, nameResultingFile):
    #This function exports to npy
    with open(routeResultingFile + nameResultingFile + ".npy", 'wb') as f:
        np.save(f, objectToSerialise)
    f.close()
    
def serialiseWordOrList(objectToSerialise, routeResultingFile, 
                        nameResultingFile):
    #This function exports to pkl
    with open(routeResultingFile + nameResultingFile + ".pkl", 'wb') as f:
        pickle.dump(objectToSerialise, f)
    f.close()
    
def deserialise(routeToFile, nameFile):
    #This function imports a npy file
    with open(routeToFile  +  nameFile + '.npy', 'rb') as f:
        temp = np.load(f, allow_pickle=True)
    f.close()
    return temp

def terminal(command):
    os.system(command)   