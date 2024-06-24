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
#  This routine trains a neural network with linear elastic data, where the 
#  strains are taken from a simulation that used the Hookean law. 

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import numpy as np
from distutils.dir_util import mkpath
from functions import *
from input_file import *
from sklearn.preprocessing import MinMaxScaler
import random
from joblib import dump
import pickle
import matplotlib.pyplot as plt

# Seed everything 
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create the scalers 
x_scaler =  MinMaxScaler()
y_scaler  =  MinMaxScaler()    


mkpath(strains_path)
mkpath(stresses_path)


withNoise = True
# withNoise = False


# Create the dataset 
# Read the expected strains
# with open('original_x_training_set.npy', 'rb') as f:
#     original_x_training_set = np.load(f)
# f.close()
# with open('original_y_training_set.npy', 'rb') as f:
#     original_y_training_set = np.load(f)
# f.close()
# shape = original_x_training_set.shape

#Add noise to the training set
# noise_weights = np.random.rand(shape[0], shape[1]) * 1.5
# original_x_training_set = np.concatenate((original_x_training_set, 
#                                           noise_weights * 
#                                           original_x_training_set), axis = 0) 


# if (withNoise):
#     # original_x_training_set = np.concatenate((original_x_training_set,
#     #                                           2 * original_x_training_set,
#     #                                           -1 * original_x_training_set,
#     #                                           -2 * original_x_training_set), 
#     #                                          axis = 0)
#     extra_strains = np.concatenate((2 * original_x_training_set,
#                                     -2 * original_x_training_set,
#                                     -1 * original_x_training_set), 
#                                     axis = 0)
#     # extra_strains[:, :] = 1 #Remove this
#     noise_weights = np.random.rand(int(extra_strains.shape[0]),
#                                    int(extra_strains.shape[1]))
#     noise_weights[:,:] = 1
#     extra_strains = noise_weights * extra_strains
#     original_x_training_set = np.concatenate((original_x_training_set,
#                                              extra_strains),
#                                              axis = 0)
#     shape = original_x_training_set.shape

if (RANDOM_STRAINS):
    # Note that write_random_strains returns the np.cumsum of the strains, not
    # the strains themselves, so that original_y_training_set can be calculated as follows
    # using the hookean law
    original_x_training_set = write_random_strains(strains_path, n_dim, number_strain_sequences, 
                 sequence_lenght, max_abs_value_deformation, type_strains)


mae = tf.keras.losses.MeanAbsoluteError()

# # Calculate stresses 
# #Initialise original_y_training_set
original_y_training_set = np.zeros(original_x_training_set.shape)

# #Populate original_y_training_set
# original_y_training_set[:, 0] = 2 * LAME_2 * original_x_training_set[:, 0] \
#     + LAME_1 * (original_x_training_set[:, 0] \
#     + original_x_training_set[:, 3] + original_x_training_set[:, 5])
        
# original_y_training_set[:, 1] = 2 * LAME_2 * original_x_training_set[:, 1]

# original_y_training_set[:, 2] = 2 * LAME_2 * original_x_training_set[:, 2]

# original_y_training_set[:, 3] = 2 * LAME_2 * original_x_training_set[:, 3] \
#     + LAME_1 * (original_x_training_set[:, 0] \
#     + original_x_training_set[:, 3] + original_x_training_set[:, 5])
        
# original_y_training_set[:, 4] = 2 * LAME_2 * original_x_training_set[:, 4]

# original_y_training_set[:, 5] = 2 * LAME_2 * original_x_training_set[:, 5] \
#     + LAME_1 * (original_x_training_set[:, 0] \
#     + original_x_training_set[:, 3] + original_x_training_set[:, 5])

#Populate original_y_training_set
original_y_training_set[:, :, 0] = 2 * LAME_2 * original_x_training_set[:, :, 0] \
    + LAME_1 * (original_x_training_set[:, :, 0] \
    + original_x_training_set[:, :, 3] + original_x_training_set[:, :, 5])
        
original_y_training_set[:, :, 1] = 2 * LAME_2 * original_x_training_set[:, :, 1]

original_y_training_set[:, :, 2] = 2 * LAME_2 * original_x_training_set[:, :, 2]

original_y_training_set[:, :, 3] = 2 * LAME_2 * original_x_training_set[:, :, 3] \
    + LAME_1 * (original_x_training_set[:, :, 0] \
    + original_x_training_set[:, :, 3] + original_x_training_set[:, :, 5])
        
original_y_training_set[:, :, 4] = 2 * LAME_2 * original_x_training_set[:, :, 4]

original_y_training_set[:, :, 5] = 2 * LAME_2 * original_x_training_set[:, :, 5] \
    + LAME_1 * (original_x_training_set[:, :, 0] \
    + original_x_training_set[:, :, 3] + original_x_training_set[:, :, 5])

x_train = original_x_training_set
x_train[:, 1:, :] = np.diff(x_train, axis = 1)
y_train = original_y_training_set


if (ML_MODEL_IS_3X3):
    x_train = np.delete(x_train, [2, 4, 5], axis=2)    
    y_train = np.delete(y_train, [2, 4, 5], axis=2)    


x_scaler.fit(x_train.reshape([x_train.shape[0] * x_train.shape[1],
                              x_train.shape[2]]))
y_scaler.fit(y_train.reshape([y_train.shape[0] * y_train.shape[1],
                              y_train.shape[2]]))

x_train_normalised = x_scaler.transform(x_train.reshape([x_train.shape[0] * x_train.shape[1],
                              x_train.shape[2]]))
y_train_normalised = y_scaler.transform(y_train.reshape([x_train.shape[0] * x_train.shape[1],
                              x_train.shape[2]]))




# x_train[:, 1:, :] = np.diff(x_train, axis = 1)
# TOTAL_LOAD_INCREMENTS = 20#12

# x_train = np.swapaxes(x_train.reshape([TOTAL_LOAD_INCREMENTS,
#                                  int(x_train.shape[0]/TOTAL_LOAD_INCREMENTS), 
#                                  int(x_train.shape[1])]), 
#                                  1, 0)

# y_train = np.swapaxes(y_train.reshape([TOTAL_LOAD_INCREMENTS,
#                                  int(y_train.shape[0]/TOTAL_LOAD_INCREMENTS), 
#                                  int(y_train.shape[1])]), 
#                                  1, 0)   

# shape = x_train.shape
# print("shape is ", shape, "!\n \n" )

# [# points in mesh, timesteps, features]
# if(withNoise):
#     x_train1 = x_train[:int(x_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     x_train1 = x_train1.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(x_train1.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(x_train1.shape[1]))
#                                )
#     y_train1 = y_train[:int(y_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     y_train1 = y_train1.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(y_train1.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(y_train1.shape[1]))
#                                )    
    
#     x_train2 = x_train[int(x_train.shape[0]/4):2*int(x_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     x_train2 = x_train2.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(x_train2.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(x_train2.shape[1]))
#                                )

#     y_train2 = x_train[int(y_train.shape[0]/4):2*int(y_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     y_train2 = y_train2.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(y_train2.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(y_train2.shape[1]))
#                                )

#     x_train3 = x_train[2 * int(x_train.shape[0]/4):3*int(x_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     x_train3 = x_train3.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(x_train3.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(x_train3.shape[1]))
#                                )

#     y_train3 = y_train[2 * int(y_train.shape[0]/4):3*int(y_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     y_train3 = y_train3.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(y_train3.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(y_train3.shape[1]))
#                                )

#     x_train4 = x_train[3 * int(x_train.shape[0]/4):4*int(x_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     x_train4 = x_train4.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(x_train4.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(x_train4.shape[1]))
#                                )
#     y_train4 = y_train[3 * int(y_train.shape[0]/4):4*int(y_train.shape[0]/4), :] # Number 4 is because there are 4 sets of strains, concatenated
#     y_train4 = y_train4.reshape((TOTAL_LOAD_INCREMENTS,
#                                  int(y_train4.shape[0]/TOTAL_LOAD_INCREMENTS),
#                                  int(y_train4.shape[1]))
#                                )
    
#     # Replace the noise
#     x_train1[:, :, 2] = 0
#     x_train1[:, :, 4] = 0
#     x_train1[:, :, 5] = 0 
#     y_train1[:, :, 2] = 0
#     y_train1[:, :, 4] = 0
    
#     x_train2[:, :, 2] = 0
#     x_train2[:, :, 4] = 0
#     x_train2[:, :, 5] = 0 
#     y_train2[:, :, 2] = 0
#     y_train2[:, :, 4] = 0
    
#     x_train3[:, :, 2] = 0
#     x_train3[:, :, 4] = 0
#     x_train3[:, :, 5] = 0 
#     y_train3[:, :, 2] = 0
#     y_train3[:, :, 4] = 0
    
#     x_train4[:, :, 2] = 0
#     x_train4[:, :, 4] = 0
#     x_train4[:, :, 5] = 0 
#     y_train4[:, :, 2] = 0
#     y_train4[:, :, 4] = 0
#     # y_train[:, :, 5] = 0 
#     ####
    
    # x_train = x_train.reshape([int(TOTAL_LOAD_INCREMENTS * 4 ), 
    #                            int(shape[0]/(TOTAL_LOAD_INCREMENTS * 4)), 
    #                            int(shape[1])])
    
    # y_train = y_train.reshape([int(TOTAL_LOAD_INCREMENTS * 4), 
    #                            int(shape[0]/(TOTAL_LOAD_INCREMENTS * 4)), 
    #                            int(shape[1])])
    
# else:
#     x_train = x_train.reshape([int(TOTAL_LOAD_INCREMENTS), 
#                                int(shape[0]/TOTAL_LOAD_INCREMENTS), 
#                                int(shape[1])])
    
#     y_train = y_train.reshape([int(TOTAL_LOAD_INCREMENTS), 
#                                int(shape[0]/TOTAL_LOAD_INCREMENTS), 
#                                int(shape[1])])

#     # Replace the noise
#     x_train[:, :, 2] = 0
#     x_train[:, :, 4] = 0
#     x_train[:, :, 5] = 0 
#     y_train[:, :, 2] = 0
#     y_train[:, :, 4] = 0
#     # y_train[:, :, 5] = 0 
#     ####

###If cubic root is applied 
# y_train = np.cbrt(y_train)
# y_train = np.cbrt(y_train/1e6)
##### 

# if (withNoise):
#     x_train1 = np.swapaxes(x_train1, 0, 1)
#     x_train1[:, 1:, :] = np.diff(x_train1, axis = 1)

#     x_train2 = np.swapaxes(x_train2, 0, 1)
#     x_train2[:, 1:, :] = np.diff(x_train2, axis = 1)
    
#     x_train3 = np.swapaxes(x_train3, 0, 1)
#     x_train3[:, 1:, :] = np.diff(x_train3, axis = 1)

#     x_train4 = np.swapaxes(x_train4, 0, 1)
#     x_train4[:, 1:, :] = np.diff(x_train4, axis = 1)
    
#     y_train1 = np.swapaxes(y_train1, 0, 1)
#     y_train2 = np.swapaxes(y_train2, 0, 1)
#     y_train3 = np.swapaxes(y_train3, 0, 1)
#     y_train4 = np.swapaxes(y_train4, 0, 1)
    
#     x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4), axis = 0)
#     y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis = 0)
    
#     x_scaler.fit(x_train.reshape([x_train.shape[0] * x_train.shape[1],
#                                   x_train.shape[2]]))
#     y_scaler.fit(y_train.reshape([y_train.shape[0] * y_train.shape[1],
#                                   y_train.shape[2]]))
    
#     x_train_normalised = x_scaler.transform(x_train.reshape([x_train.shape[0] * x_train.shape[1],
#                                   x_train.shape[2]]))
#     y_train_normalised = y_scaler.transform(y_train.reshape([x_train.shape[0] * x_train.shape[1],
#                                   x_train.shape[2]]))
    
# x_train_normalised = x_train_normalised.reshape(x_train.shape)
# y_train_normalised = y_train_normalised.reshape(y_train.shape)
    
# else:
    

#     x_train = np.swapaxes(x_train, 0, 1)
#     x_train[:, 1:, :] = np.diff(x_train, axis = 1)
    
#     y_train = np.swapaxes(y_train, 0, 1)


#     x_scaler.fit(x_train.reshape(shape))
#     y_scaler.fit(y_train.reshape(shape))

#     x_train_normalised = x_scaler.transform(x_train.reshape(shape))
#     y_train_normalised = y_scaler.transform(y_train.reshape(shape))
    
x_train_normalised = x_train_normalised.reshape(x_train.shape)
y_train_normalised = y_train_normalised.reshape(y_train.shape)
        
if (SUBSAMPLE_ORIGINAL_STRAINS):
    idx = np.random.randint(0, int(x_train.shape[0]), 
                            size = int(1 * x_train.shape[0]))

# Create the ML_Model 
# ML_model = createNN(ML_MODEL_IS_3X3, 7)
ML_model = createRNN(ML_MODEL_IS_3X3, 10)
compileNN(ML_model, False)       
print(ML_model.summary())


# history = ML_model.fit(x_train_normalised[idx, :, :], 
#                y_train_normalised[idx, :, :], 
#                 epochs = NUMBER_OF_EPOCHS, 
#                 validation_split=0.01)

# if (ML_MODEL_IS_3X3):
#     x = np.delete(x_train_normalised, [2, 4, 5], 2)
#     y = np.delete(y_train_normalised, [2, 4], 2)
#     history = ML_model.fit(x[idx, :, :], 
#                     y[idx, :, :], 
#                     epochs = NUMBER_OF_EPOCHS)    

    
# else:

#     history = ML_model.fit(x_train_normalised[idx, : :], 
#                     y_train_normalised[idx, :, :], 
#                     # epochs = NUMBER_OF_EPOCHS)
#                     # epochs = 20000, validation_split = 0.05)
#                     epochs = 20000, validation_split = 0.1)
    
    
history = ML_model.fit(x_train_normalised[idx, : :], 
                y_train_normalised[idx, :, :], 
                # epochs = NUMBER_OF_EPOCHS)
                # epochs = 20000, validation_split = 0.05)
                epochs = 20000, validation_split = 0.1)
    
    
    # exit()

    # history = ML_model.fit(x_train_normalised[idx, :15, :], 
    #                 y_train_normalised[idx, :15, :], 
    #                 # epochs = NUMBER_OF_EPOCHS)
    #                 epochs = 100000)
    
    # history = ML_model.fit(x_train_normalised[idx, :15, :], 
    #                 y_train_normalised[idx, :15, :], 
    #                 # epochs = NUMBER_OF_EPOCHS)
    #                 epochs = 50000)
    
    
    # history = ML_model.fit(x_train_normalised[idx, :30, :], 
    #                 y_train_normalised[idx, :30, :], 
    #                 epochs = NUMBER_OF_EPOCHS)
    
    # history = ML_model.fit(x_train_normalised[idx, :, :], 
    #                 y_train_normalised[idx, :, :], 
    #                 epochs = 200000)
    
    # history = ML_model.fit(x_train_normalised[idx, :15, :], 
    #                 y_train_normalised[idx, :15, :], 
    #                 # epochs = 120000)
    #                 epochs = 100000)
    # history = ML_model.fit(x_train_normalised[idx, :20, :], 
    #                 y_train_normalised[idx, :20, :], 
    #                 # epochs = 120000)
    #                 epochs = 100000)
    # history = ML_model.fit(x_train_normalised[idx, :25, :], 
    #                 y_train_normalised[idx, :25, :], 
    #                 # epochs = 120000)
    #                 epochs = 100000)
    # history = ML_model.fit(x_train_normalised[idx, :30, :], 
    #                 y_train_normalised[idx, :30, :], 
    #                 # epochs = 120000)
    #                 epochs = 100000)
    
    # x = np.delete(x_train_normalised, [2, 4, 5], 2)
    # y = np.delete(y_train_normalised, [2, 4, 5], 2)
    # history = ML_model.fit(x[idx, :, :], 
    #                 y[idx, :, :], 
    #                 epochs = NUMBER_OF_EPOCHS)    


# exit()

# x_train = x_train[idx, :]
# y_train = y_train[idx, :]

# Save the ML model
ML_model.save("ML_model.h5")

# Serialise the dataset and the scalers
with open('x_train.npy', 'wb') as f:
    pickle.dump(x_train, f)    
f.close()
with open('y_train.npy', 'wb') as f:
    pickle.dump(y_train, f)
f.close()

with open('x_train_scaled.npy', 'wb') as f:
    pickle.dump(x_train_normalised, f)    
f.close()
with open('y_train_scaled.npy', 'wb') as f:
    pickle.dump(y_train_normalised, f)
f.close()


# # Serialise the dataset and the scalers
# with open('x_train.npy', 'wb') as f:
#     pickle.dump(x_train[:, :5, :], f)    
# f.close()
# with open('y_train.npy', 'wb') as f:
#     pickle.dump(y_train[:, :5, :], f)
# f.close()

# with open('x_train_scaled.npy', 'wb') as f:
#     pickle.dump(x_train_normalised[:, :5, :], f)    
# f.close()
# with open('y_train_scaled.npy', 'wb') as f:
#     pickle.dump(y_train_normalised[:, :5, :], f)
# f.close()


# Serialise the scalers
dump(x_scaler, 'x_scaler.joblib')
dump(y_scaler, 'y_scaler.joblib')
###############################################################################

# exit()
# # postproc 


y_pn = ML_model.predict(x_train_normalised)
y_p = y_scaler.inverse_transform(y_pn.reshape([y_pn.shape[0] * y_pn.shape[1], 
                                                y_pn.shape[2]]))
y_p = y_p.reshape(y_train.shape)
# error = mae(y_train[:, :, :], y_p[:, :, :])
error = mae(y_train[:, :15, :], y_p[:, :15, :])
# error = mae(y_train[:, :15, :]**3, y_p[:, :15, :]**3)
# error = mae((y_train[:, :15, :]**3)*1e6, (y_p[:, :15, :]**3)*1e6)
print(error)



def plotResults(x, y, y_pred, plots_path, history, number_of_plots, n_dim):
    places = np.arange(number_of_plots)
    mkpath(plots_path)
    # labels = ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", 
    #           "sigma_xz",  "sigma_yz"]
    
    if (n_dim == 2):
        labels = ["sigma_xx", "sigma_xy", "sigma_yy"]
        labels_pred = ["sigma_xx_pred", "sigma_xy_pred", "sigma_yy_pred"]
        labels_colors = ["dimgray", "steelblue", "forestgreen"]
        labels_colors_predictions = ["lightgray", "deepskyblue", "lime", 
                                      "lightcoral", "peru", "darkviolet", 
                                      "yellow"]
    else:
        labels = ["sigma_xx", "sigma_xy", "sigma_xz", "sigma_yy", 
              "sigma_yz",  "sigma_zz"]
    # labels_pred = ["sigma_xx_pred", "sigma_yy_pred", "sigma_zz_pred", 
    #              "sigma_xy_pred", "sigma_xz_pred", "sigma_yz_pred"]
        labels_pred = ["sigma_xx_pred", "sigma_xy_pred", "sigma_xz_pred", 
                      "sigma_yy_pred", "sigma_yz_pred", "sigma_zz_pred"]
        labels_colors = ["dimgray", "steelblue", "forestgreen", 
                          "indianred", "saddlebrown", "purple",  "gold"]
        labels_colors_predictions = ["lightgray", "deepskyblue", "lime", 
                                      "lightcoral", "peru", "darkviolet", 
                                      "yellow"]
    # coord_x = np.array(range(0, x.shape[1] + 1), dtype = float)
    coord_x = np.array(range(x.shape[1]), dtype = float)
    for i in places:
        coord_y = np.zeros(2)            
        coord_y_pred = np.zeros(2)
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)
        
        plot_name  = plots_path + '/test_sample_number_' + str(i) + '.png'
        if (n_dim == 2):
            sup_ind = 3
            comp_to_plot = 2
            
        else:
            sup_ind = 6
            comp_to_plot = 3
        
        for j in range(sup_ind):
            # coord_y[1] = y[i, :, j]
            # coord_y_pred[1] = y_pred[i, :, j]   
            coord_y = y[i, :, j]
            coord_y_pred = y_pred[i, :, j]   
            # if (j in [3]): #Do not plot unwanted quantities
            # if (j in [0, 1, 3, 5]): #Do not plot unwanted quantities
            if (j in [comp_to_plot]): #Do not plot unwanted quantities
                # plt.plot(coord_x, coord_y, label = labels[j], color = 
                #           labels_colors[j], alpha = 0.6)
                # plt.plot(coord_x, coord_y_pred, linestyle='dashed', label = 
                #           labels_pred[j], color = labels_colors[j])  

                plt.plot(coord_x[:15], coord_y[:15], label = labels[j], color = 
                          labels_colors[j], alpha = 0.6)
                plt.plot(coord_x[:15], coord_y_pred[:15], linestyle='dashed', label = 
                          labels_pred[j], color = labels_colors[j])


                # plt.plot(coord_x[:15], (coord_y[:15]**3), label = labels[j], color = 
                #           labels_colors[j], alpha = 0.6)
                # plt.plot(coord_x[:15], (coord_y_pred[:15]**3), linestyle='dashed', label = 
                #           labels_pred[j], color = labels_colors[j])                                       


                # plt.plot(coord_x[:15], (coord_y[:15]**3)*1e6, label = labels[j], color = 
                #           labels_colors[j], alpha = 0.6)
                # plt.plot(coord_x[:15], (coord_y_pred[:15]**3)*1e6, linestyle='dashed', label = 
                #           labels_pred[j], color = labels_colors[j])                                       

                # plt.plot(coord_x[:15], (coord_y[:15]**3)*1e6, label = labels[j], color = 
                #           labels_colors[j], alpha = 0.6)
                # plt.plot(coord_x[:15], (coord_y_pred[:15]**3)*1e6, linestyle='dashed', label = 
                #           labels_pred[j], color = labels_colors[j])   

        box = ax.get_position() #
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("time steps")
        plt.ylabel("Sigma")
        plt.grid(True)
        plt.savefig(plot_name)
        plt.close()
    pd.DataFrame(history.history).plot(figsize=(15, 10))
    plot_name = plots_path + '/Convergence history.png'
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(plot_name)
    
plotResults(x_train, y_train, y_p, PLOTS_PATH, history, 130, n_dim)

