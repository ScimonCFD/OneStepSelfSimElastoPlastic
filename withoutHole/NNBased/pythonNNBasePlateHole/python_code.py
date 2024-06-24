import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disable tf warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from joblib import dump, load
import sklearn
import pickle
from tensorflow.keras import Sequential, Model
from sklearn.preprocessing import MinMaxScaler

#Supress warnings
import warnings
warnings.filterwarnings("ignore")
####

ML_model= keras.models.load_model("ML_model.h5")

# #Load the scalers
x_scaler = load('x_scaler.joblib')
y_scaler = load('y_scaler.joblib')

# nStates = int(20) #This variable should be read from a file
# nStates = int(20) #This variable should be read from a file
nStates = int(10) #This variable should be read from a file
# time = 0

# #Read the variable "TypeMLModel"
# with open('constant/nStates.pkl', 'rb') as f:
#     TypeMLModel = pickle.load(f)
# f.close()

# with open(routeResultingFile + nameResultingFile + ".npy", 'wb') as f:
#     np.save(str(int(time))+'/epsilon', epsilon)
# f.close()


ML_MODEL_IS_3X3 = True


def serialise_fields():
    np.save(str(int(time))+'/epsilon', epsilon)
    np.save(str(int(time))+'/sigma', sigma)   
    np.save(str(int(time))+'/D', D)
    for i in range(int(nStates)):
        dumString = "states" + str(i)
        # print(np.save(str(int(time))+'/' + dumString, dumString))
        exec("np.save(str(int(time))+'/' + dumString, dumString)")
        # np.save(str(int(time))+'/states' + str(i), states)  

    np.save(str(int(time))+'/epsilon_Right', epsilon_Right)
    np.save(str(int(time))+'/epsilon_Down', epsilon_Down)
    np.save(str(int(time))+'/epsilon_Up', epsilon_Up)

    np.save(str(int(time))+'/sigma_Right', sigma_Right)
    np.save(str(int(time))+'/sigma_Down', sigma_Down)
    np.save(str(int(time))+'/sigma_Up', sigma_Up)

    # np.save(str(int(time))+'/D_Right', D_Right)
    # np.save(str(int(time))+'/D_Down', D_Down)
    # np.save(str(int(time))+'/D_Up', D_Up)



def sigmoid(x):
    return 1/(1 + np.exp(-x))

# def sigmoid(value):
#     if -value > np.log(np.finfo(type(value)).max):
#         return 0.0    
#     a = np.exp(-value)
#     return 1.0/ (1.0 + a)

def ReLU(x):
    return(np.maximum(0, x))

def pullParametersNeuralNetwork(ML_model):
    ################# Retrieve GRU parameters ####################
    GRU_Layer = ML_model.layers[0]
    nNeuronsGRU = GRU_Layer.units
    weights_GRU_Layer = GRU_Layer.get_weights()
    # https://stackoverflow.com/questions/72809642/how-to-interpret-get-weights-for-keras-gru
    W_full_matrix = weights_GRU_Layer[0] #THe so-called Kernel
    U_full_matrix = weights_GRU_Layer[1] #THe so-called Kernel
    b_full_matrix = weights_GRU_Layer[2] #THe so-called Kernel
    
    Wz = W_full_matrix[:, :nNeuronsGRU]
    Wr = W_full_matrix[:, nNeuronsGRU:nNeuronsGRU * 2]
    Wh = W_full_matrix[:, nNeuronsGRU * 2:]
    
    Uz = U_full_matrix[:, :nNeuronsGRU]
    Ur = U_full_matrix[:, nNeuronsGRU:nNeuronsGRU * 2]
    Uh = U_full_matrix[:, nNeuronsGRU * 2:]
    
    input_bias = b_full_matrix[0, :][np.newaxis, :]
    recurrent_bias = b_full_matrix[1, :][np.newaxis, :]
    
    input_bias_z = input_bias[:, :nNeuronsGRU]
    input_bias_r = input_bias[:, nNeuronsGRU:nNeuronsGRU * 2]
    input_bias_h = input_bias[:, nNeuronsGRU * 2:]
    
    recurrent_bias_z = recurrent_bias[:, :nNeuronsGRU]
    recurrent_bias_r = recurrent_bias[:, nNeuronsGRU:nNeuronsGRU * 2]
    recurrent_bias_h = recurrent_bias[:, nNeuronsGRU * 2:]
    ##############################################################


    ########## Retrieve Dense Middle Layer parameters ############
    Dense_Middle_Layer = ML_model.layers[1]
    nNeuronsDense_Middle_Layer = Dense_Middle_Layer.units
    weights_Dense_Middle_Layer = Dense_Middle_Layer.get_weights()

    WDenseMiddleLayer = weights_Dense_Middle_Layer[0]
    biasDenseMiddleLayer = weights_Dense_Middle_Layer[1]
    ##############################################################


    ########## Retrieve Dense Output Layer parameters ############
    Dense_Output_Layer = ML_model.layers[2]
    nNeuronsDense_Output_Layer = Dense_Output_Layer.units
    weights_Dense_Output_Layer = Dense_Output_Layer.get_weights()
    
    WDense_Output_Layer = weights_Dense_Output_Layer[0]
    biasDense_Output_Layer = weights_Dense_Output_Layer[1]
    ##############################################################    
    
    return Wz, Wr, Wh, Uz, Ur, Uh, input_bias_z, input_bias_r, \
           input_bias_h, recurrent_bias_z, recurrent_bias_r, \
           recurrent_bias_h, WDenseMiddleLayer, biasDenseMiddleLayer, \
           WDense_Output_Layer, biasDense_Output_Layer
           

def GRULayerNumPy(x, states):
    x_z = x.dot(Wz)
    x_r = x.dot(Wr)
    x_h = x.dot(Wh)
    
    x_z = x_z + input_bias_z
    x_r = x_r + input_bias_r
    x_h = x_h + input_bias_h    
    
    h_tm1_z = states
    h_tm1_r = states
    h_tm1_h = states
    
    # print("h_tm1_z shape is ", h_tm1_z.shape)
    # print("Uz shape is ", Uz.shape)

    recurrent_z = h_tm1_z.dot(Uz)
    recurrent_r = h_tm1_r.dot(Ur)
    
    recurrent_z += recurrent_bias_z
    recurrent_r += recurrent_bias_r
    
    z = sigmoid(x_z + recurrent_z)
    r = sigmoid(x_r + recurrent_r)
    
    recurrent_h = h_tm1_h.dot(Uh)
    recurrent_h += recurrent_bias_h
    recurrent_h = r * recurrent_h
    
    hh = np.tanh(x_h + recurrent_h)
    
    h = z * states + (1 - z) * hh

    return h


def DenseMiddleLayer(x):
    h_Middle_Layer = ReLU(x.dot(WDenseMiddleLayer) + biasDenseMiddleLayer)
    return h_Middle_Layer

def DenseOutputLayer(x):
    h_Output_Layer = (x.dot(WDense_Output_Layer) + biasDense_Output_Layer)
    return h_Output_Layer


ML_model= keras.models.load_model("ML_model.h5")




# Pull the parameters from the NN
[Wz, Wr, Wh, Uz, Ur, Uh, input_bias_z, input_bias_r, \
       input_bias_h, recurrent_bias_z, recurrent_bias_r, \
       recurrent_bias_h, WDenseMiddleLayer, biasDenseMiddleLayer, \
       WDense_Output_Layer, biasDense_Output_Layer] = pullParametersNeuralNetwork(ML_model)







def predict():
    global time
    # Remove noise
    epsilon[:, 2] = 0
    epsilon[:, 4] = 0
    epsilon[:, 5] = 0
    # print(dir())
    # print(globals())
    # print(int(time))
    # exit()
    # time = int(time)
    # if (time == 1):
    statesNN = np.zeros([int(nStates), int(epsilon.shape[0]), int(1)]) #Master array with all the states

    # else:
    #     statesNN = np.zeros([int(nStates), int(epsilon.shape[0]), int(1)])
    #     for i in range(nStates):
    #         word = states_ + str(i)
    #         exec("statesNN[i, :, :] = states_ +") 

    
    # print("1statesNN.shape is ", statesNN.shape)
    # print"("statesNN.shape ", is \n", statesNN.shape, "\n")

    #Merge all the states into one master array
    for i in range(int(nStates)):
        dumString = "states" + str(i)
        exec("statesNN[" + str(i) + ", :, :] = " + dumString)
    # print(statesNN.shape)
    # print(statesNN[1, :, :])
    # exit()

    #Rotate the master array 
    statesNN = np.rot90(statesNN, 1, (0,2))
    # print(statesNN.shape)
    # exit()

    # print("\n \n 2statesNN.shape is ", statesNN.shape)

    #Reshape to get rid of the extra axes
    statesNN = statesNN.reshape([statesNN.shape[1], statesNN.shape[2]])
    # print(statesNN.shape)
    # print(statesNN)
    # exit()

    # print("\n \n 3statesNN.shape is ", statesNN.shape)
    
    if (ML_MODEL_IS_3X3):
        epsilon_scaled = x_scaler.transform(np.delete(DEpsilon, [2,4,5], axis = 1))
        
    else:
        epsilon_scaled = x_scaler.transform(DEpsilon)

    # print(epsilon_scaled[np.argwhere(epsilon_scaled[:, :] > 1)], "\n", "\n", "\n")
    # print(epsilon_scaled[np.argwhere(epsilon_scaled[:, :] < -0.0000000001)], "\n", "\n", "\n")
    # print([np.argwhere(epsilon_scaled[:, :] > 1)], "\n", "\n", "\n")
    # print([np.argwhere(epsilon_scaled[:, :] < 0)], "\n", "\n", "\n")
    # print(epsilon_scaled[np.argwhere(epsilon_scaled[:, :] > 1)].shape, "\n", "\n", "\n")
    # print(epsilon_scaled[np.argwhere(epsilon_scaled[:, :] < 0)].shape, "\n", "\n", "\n")


    yGRU_scaled = GRULayerNumPy(epsilon_scaled, statesNN)
    

    # Output from Dense Middle layer
    yDML = DenseMiddleLayer(yGRU_scaled)

    # Output from output layer
    sigma_scaled = DenseOutputLayer(yDML)

    # print("sigma_scaled shpae is", sigma_scaled.shape)

#            sigma[:, :] = y_scaler.inverse_transform(sigma_scaled.reshape(epsilon_scaled.shape))
    
    if (ML_MODEL_IS_3X3):
        sigma_calc = ((y_scaler.inverse_transform(sigma_scaled.reshape(epsilon_scaled.shape))))
        sigma[:, :] = 0
        sigma[:, 0:2] = sigma_calc[:, 0:2]
        sigma[:, 3] = sigma_calc[:, 2]

    else:
        sigma[:, :] = ((y_scaler.inverse_transform(sigma_scaled.reshape(epsilon_scaled.shape))))

    # If additinoal transformations
    # sigma[:, :] = (sigma**3)
    

    # Remove noise
    sigma[:, 2] = 0
    sigma[:, 4] = 0
    # sigma[:, 5] = 0


    #Repopulate the dumStates
    statesNN = yGRU_scaled
    # print("\n \n 4statesNN.shape is ", statesNN.shape)

    #Merge all the states into one master array
    for i in range(int(nStates)):
        dumString = "dumStates" + str(i)
        # print("\n", dumStates1.shape)
        # print(dumString + "[:, :] = yGRU_scaled[:, " + str(i) + "].reshape(" + dumString + ".shape)")
        exec(dumString + "[:, :] = yGRU_scaled[:, " + str(i) + "].reshape(" + dumString + ".shape)")

    # print("\n \n \n time is ", time)