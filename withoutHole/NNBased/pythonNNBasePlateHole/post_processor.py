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
from joblib import dump
from joblib import load
from distutils.dir_util import mkpath
import matplotlib.pyplot as plt

mkpath("./ComparisonResults/")
mkpath("./ComparisonResults/WithBoundaryNodes")
mkpath("./ComparisonResults/WithoutBoundaryNodes")
mkpath("./ComparisonResults/epsilon_history/")
mkpath("./ComparisonResults/sigma_history/")

def deserialise(routeToFile, nameFile):
    #This function imports a npy file
    with open(routeToFile  +  nameFile + '.npy', 'rb') as f:
        temp = np.load(f, allow_pickle=True)
    f.close()
    return temp


endTime = 10
ROUTE_THEORETICAL_MODEL = "./EP_Theoretical/"
ROUTE_THEORETICAL_ELASTIC_MODEL = "./fullyElastic/"
ROUTE_EP_NN_MODEL = "./With_EP_NN/"
ROUTE_ELASTIC_NN_MODEL = "./With_Elastic_NN/" 

# Set the default text font size
plt.rc('font', size=16)# Set the axes title font size
plt.rc('axes', titlesize=20)# Set the axes labels font size
plt.rc('axes', labelsize=20)# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)# Set the legend font size
plt.rc('legend', fontsize=18)# Set the font size of the figure title
plt.rc('figure', titlesize=24)

for load_inc in range(1, endTime+1):  
    # Elasto-plastic theoretical results
    epsilon_Right = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "epsilon_Right")
    epsilon_Down = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "epsilon_Down")
    epsilon_Up = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "epsilon_Up")  
    # epsilon_Hole = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
    #                                       "/", "epsilon_Hole")  
    sigma_Right = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "sigma_Right")
    sigma_Down = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "sigma_Down")
    sigma_Up = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "sigma_Up")
    # sigma_Hole = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
    #                                       "/", "sigma_Hole")  

    # Elasto-plastic NN results
    ep_eps_Right = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                              "epsilon_Right")
    ep_eps_Down = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                              "epsilon_Down")
    ep_eps_Up = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                            "epsilon_Up")
    # ep_eps_Hole = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
    #                           "epsilon_Hole")
    
    ep_sig_Right = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                               "sigma_Right")
    ep_sig_Down = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                              "sigma_Down")
    ep_sig_Up = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                            "sigma_Up")
    # ep_sig_Hole = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
    #                           "sigma_Hole")

    

    # Elastic theoretical results
    elastic_eps_Right = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                    str(int(load_inc)) + "/", "epsilon_Right")
    elastic_eps_Down = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                   str(int(load_inc)) + "/", "epsilon_Down")
    elastic_eps_Up = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                 str(int(load_inc)) + "/", "epsilon_Up")
    # elastic_eps_Hole = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
    #                                str(int(load_inc)) + "/", "epsilon_Hole")
    
    
    elastic_sig_Right = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                    str(int(load_inc)) + "/", "sigma_Right")
    elastic_sig_Down = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                   str(int(load_inc)) + "/", "sigma_Down")
    elastic_sig_Up = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                 str(int(load_inc)) + "/", "sigma_Up")
    # elastic_sig_Hole = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
    #                                str(int(load_inc)) + "/", "sigma_Hole")


    # Elastic NN results
    elasticNN_eps_Right = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                      str(int(load_inc)) + "/", 
                                      "epsilon_Right")
    elasticNN_eps_Down = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                     str(int(load_inc)) + "/", "epsilon_Down")
    elasticNN_eps_Up = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                   str(int(load_inc)) + "/", "epsilon_Up")
    # elasticNN_eps_Hole = deserialise(ROUTE_ELASTIC_NN_MODEL 
    #                                  + str(int(load_inc)) + "/", 
    #                                  "epsilon_Hole")
    
    elasticNN_sig_Right = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                      str(int(load_inc)) + "/", "sigma_Right")
    elasticNN_sig_Down = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                     str(int(load_inc)) + "/", "sigma_Down")
    elasticNN_sig_Up = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                   str(int(load_inc)) + "/", "sigma_Up")
    # elasticNN_sig_Hole = deserialise(ROUTE_ELASTIC_NN_MODEL + 
    #                                  str(int(load_inc)) + "/", "sigma_Hole")    

    if(load_inc == 1):
        
        theoretical_ep_epsilon_field = deserialise(ROUTE_THEORETICAL_MODEL + 
                                                   str(int(load_inc)) + "/", 
                                                   "epsilon")
        theoretical_ep_sigma_field = deserialise(ROUTE_THEORETICAL_MODEL + 
                                                 str(int(load_inc)) + "/", 
                                                 "sigma")
        
        ep_nn_epsilon_field = deserialise(ROUTE_EP_NN_MODEL + 
                                          str(int(load_inc)) + "/", "epsilon")
        ep_nn_sigma_field = deserialise(ROUTE_EP_NN_MODEL + 
                                        str(int(load_inc)) + "/", "sigma")
        
        theoretical_elastic_epsilon_field = deserialise(
                                              ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                              str(int(load_inc)) + "/", 
                                              "epsilon")
        theoretical_elastic_sigma_field = deserialise(
                                              ROUTE_THEORETICAL_ELASTIC_MODEL +  
                                              str(int(load_inc)) + "/", 
                                              "sigma")
        
        elastic_nn_epsilon_field = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                              str(int(load_inc)) + "/", 
                                              "epsilon")

        elastic_nn_sigma_field = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                             str(int(load_inc)) + "/", "sigma")
        
        
    else:
        
        theoretical_ep_epsilon_field_current = deserialise(
                                                      ROUTE_THEORETICAL_MODEL +
                                                      str(int(load_inc)) + "/", 
                                                      "epsilon")
        theoretical_ep_epsilon_field = np.concatenate((
                                         theoretical_ep_epsilon_field, 
                                         theoretical_ep_epsilon_field_current),
                                         axis=0)    
        
        ep_nn_epsilon_field_current = deserialise(ROUTE_EP_NN_MODEL + 
                                                  str(int(load_inc)) + "/", 
                                                  "epsilon")
        ep_nn_epsilon_field = np.concatenate((ep_nn_epsilon_field, 
                                              ep_nn_epsilon_field_current), 
                                              axis=0)          
        
        theoretical_elastic_epsilon_field_current = deserialise(
                                             ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                             str(int(load_inc)) + "/", 
                                             "epsilon")
        theoretical_elastic_epsilon_field = np.concatenate((
                                    theoretical_elastic_epsilon_field , 
                                    theoretical_elastic_epsilon_field_current),
                                    axis=0)    

        elastic_nn_epsilon_field_current = deserialise(ROUTE_ELASTIC_NN_MODEL +
                                                       str(int(load_inc)) + 
                                                       "/", "epsilon")
        elastic_nn_epsilon_field = np.concatenate((
                                             elastic_nn_epsilon_field, 
                                             elastic_nn_epsilon_field_current), 
                                             axis=0)          
        
        
        theoretical_ep_sigma_field_current = deserialise(
                                 ROUTE_THEORETICAL_MODEL + str(int(load_inc)) +
                                    "/", "sigma")
        theoretical_ep_sigma_field = np.concatenate((
                                           theoretical_ep_sigma_field, 
                                           theoretical_ep_sigma_field_current),
                                           axis=0)    
        
        ep_nn_sigma_field_current = deserialise(ROUTE_EP_NN_MODEL + 
                                                str(int(load_inc)) + "/", 
                                                "sigma")
        ep_nn_sigma_field = np.concatenate((ep_nn_sigma_field, 
                                            ep_nn_sigma_field_current), 
                                            axis=0)          
        
        theoretical_elastic_sigma_field_current = deserialise(
                                              ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                              str(int(load_inc)) + "/", 
                                              "sigma")
        theoretical_elastic_sigma_field = np.concatenate((
                                      theoretical_elastic_sigma_field , 
                                      theoretical_elastic_sigma_field_current), 
                                      axis=0)    

        elastic_nn_sigma_field_current = deserialise(ROUTE_ELASTIC_NN_MODEL + 
                                                     str(int(load_inc)) + "/", 
                                                     "sigma")
        elastic_nn_sigma_field = np.concatenate((
                                               elastic_nn_sigma_field, 
                                               elastic_nn_sigma_field_current),
                                               axis=0)          
        
        
        
    # Theoretical elastoplastic
    theoretical_ep_epsilon_field = np.concatenate((theoretical_ep_epsilon_field, 
                                                   epsilon_Right, epsilon_Down,
                                                   epsilon_Up), 
                                                   axis = 0)
    
    theoretical_ep_sigma_field = np.concatenate((theoretical_ep_sigma_field,  
                                                sigma_Right, sigma_Down,
                                                sigma_Up), 
                                                axis = 0)
    
    # NN elastoplastic
    ep_nn_epsilon_field = np.concatenate((ep_nn_epsilon_field, ep_eps_Right,
                                          ep_eps_Down, ep_eps_Up), 
                                          axis = 0)

    ep_nn_sigma_field = np.concatenate((ep_nn_sigma_field, ep_sig_Right,
                                        ep_sig_Down, ep_sig_Up), 
                                        axis = 0)
    
    # Theoretical elastic
    theoretical_elastic_epsilon_field = np.concatenate((
                                             theoretical_elastic_epsilon_field, 
                                             elastic_eps_Right, 
                                             elastic_eps_Down, elastic_eps_Up), 
        axis = 0)
    
    theoretical_elastic_sigma_field = np.concatenate((
                                              theoretical_elastic_sigma_field,  
                                              elastic_sig_Right,
                                              elastic_sig_Down,
                                              elastic_sig_Up), axis = 0)
    
    # NN Elastic
    elastic_nn_epsilon_field = np.concatenate((elastic_nn_epsilon_field, 
                                              elasticNN_eps_Right,
                                              elasticNN_eps_Down,
                                              elasticNN_eps_Up), axis = 0)

    elastic_nn_sigma_field = np.concatenate((elastic_nn_sigma_field,  
                                             elasticNN_sig_Right,
                                             elasticNN_sig_Down,
                                             elasticNN_sig_Up), axis = 0)
    
    
    

shape = theoretical_ep_epsilon_field.shape
theoretical_ep_epsilon_field = theoretical_ep_epsilon_field.reshape([
                                                         int(endTime), 
                                                         int(shape[0]/endTime), 
                                                         shape[1]])

        
theoretical_ep_sigma_field = theoretical_ep_sigma_field.reshape([
                                                         int(endTime), 
                                                         int(shape[0]/endTime), 
                                                         shape[1]])


ep_nn_epsilon_field = ep_nn_epsilon_field.reshape([int(endTime), 
                                                   int(shape[0]/endTime), 
                                                   shape[1]])

ep_nn_sigma_field = ep_nn_sigma_field.reshape([int(endTime), 
                                               int(shape[0]/endTime), 
                                               shape[1]])


theoretical_elastic_epsilon_field = theoretical_elastic_epsilon_field.reshape([
                                                         int(endTime), 
                                                         int(shape[0]/endTime),
                                                         shape[1]])

theoretical_elastic_sigma_field = theoretical_elastic_sigma_field.reshape([
                                                         int(endTime), 
                                                         int(shape[0]/endTime),
                                                         shape[1]])

elastic_nn_epsilon_field = elastic_nn_epsilon_field.reshape([
                                                         int(endTime), 
                                                         int(shape[0]/endTime), 
                                                         shape[1]])

elastic_nn_sigma_field = elastic_nn_sigma_field.reshape([int(endTime), 
                                                         int(shape[0]/endTime), 
                                                         shape[1]])



component = ["xx", "xy", "xz", "yy", "yz", "zz"]
dum_ind = (0, 1, 3)
for com in range(len(component)):            
    # if (com in (0, 1, 3)):
        if (com in dum_ind):
            fig = plt.figure(figsize=(15, 10))
            # if (ML_MODEL_IS_3X3):
            plt.plot(theoretical_ep_epsilon_field[-1, :, :][:, com], 
                     theoretical_ep_epsilon_field[-1, :, :][:, com], 
                     color = "blue", label = "Expected", marker='x')                
            plt.scatter(theoretical_ep_epsilon_field[-1, :, :][:, com], 
                        elastic_nn_epsilon_field[-1, :, :][:, com], 
                        color = "red", label = "ElasticNN", marker='o', 
                        alpha = 0.3)                
            plt.scatter(theoretical_ep_epsilon_field[-1, :, :][:, com], 
                        theoretical_elastic_epsilon_field[-1, :, :][:, com], 
                        color = "purple", label = "ElasticTheoretical", 
                        marker='+')   
            plt.scatter(theoretical_ep_epsilon_field[-1, :, :][:, com], 
                        ep_nn_epsilon_field[-1, :, :][:, com], 
                        color = "green", label = "Calculated", marker='o', 
                        alpha=(0.3))          
            plt.xlabel(r'Expected strain $(m/m)$')
            plt.ylabel(r'Calculated strain $(m/m)$')
            # fig.suptitle("Epsilon_" + component[com], fontsize=20)
            # plt.legend()
            # plt.show()
            fig.savefig("ComparisonResults/WithBoundaryNodes" +"/epsilon_" + 
                        component[com] + ".png", bbox_inches='tight')
            plt.close(fig)  
            
            fig = plt.figure(figsize=(15, 10))
            plt.plot(theoretical_ep_epsilon_field[-1, :, :][:100, com], 
                     theoretical_ep_epsilon_field[-1, :, :][:100, com], 
                     color = "blue", label = "Expected", marker='x')                
            plt.scatter(theoretical_ep_epsilon_field[-1, :, :][:100, com], 
                        elastic_nn_epsilon_field[-1, :, :][:100, com], 
                        color = "red", label = "ElasticNN", marker='o', 
                        alpha = 0.3)                
            plt.scatter(theoretical_ep_epsilon_field[-1, :, :][:100, com], 
                        theoretical_elastic_epsilon_field[-1, :, :][:100, com], 
                        color = "purple", label = "ElasticTheoretical", 
                        marker='+')   
            plt.scatter(theoretical_ep_epsilon_field[-1, :, :][:100, com], 
                        ep_nn_epsilon_field[-1, :, :][:100, com], 
                        color = "green", label = "Calculated", marker='o', 
                        alpha=(0.3))  
            plt.xlabel(r'Expected strain $(m/m)$')
            plt.ylabel(r'Calculated strain $(m/m)$')
            # fig.suptitle("Epsilon_" + component[com], fontsize=20)
            # plt.legend()
            # plt.show()
            fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/epsilon_" + 
                        component[com] + ".png", bbox_inches='tight')
            plt.close(fig)            
            
            
for com in range(len(component)):            
    # if (com in (0, 1, 3)):
        if (com in dum_ind):
            fig = plt.figure(figsize=(15, 10))
            # if (ML_MODEL_IS_3X3):
            plt.plot(theoretical_ep_sigma_field[-1, :, :][:, com], 
                    theoretical_ep_sigma_field[-1, :, :][:, com], 
                    color = "blue", label = "Expected", marker='x')                
            plt.scatter(theoretical_ep_sigma_field[-1, :, :][:, com], 
                        elastic_nn_sigma_field[-1, :, :][:, com], 
                        color = "red", label = "ElasticNN", marker='o', 
                        alpha = 0.3)                
            plt.scatter(theoretical_ep_sigma_field[-1, :, :][:, com], 
                        theoretical_elastic_sigma_field[-1, :, :][:, com],
                        color = "purple", label = "ElasticTheoretical",
                        marker='+')   
            plt.scatter(theoretical_ep_sigma_field[-1, :, :][:, com],
                        ep_nn_sigma_field[-1, :, :][:, com], color = "green", 
                        label = "Calculated", marker='o', alpha=(0.3))  
            plt.xlabel(r'Expected stress $(Pa)$')
            plt.ylabel(r'Calculated stress $(Pa)$')
            # fig.suptitle("Sigma_" + component[com], fontsize=20)
            # plt.legend()
            # plt.show()
            fig.savefig("ComparisonResults/WithBoundaryNodes" + "/sigma_" + 
                        component[com] + ".png", bbox_inches='tight')
            plt.close(fig)
        
            fig = plt.figure(figsize=(15, 10))
            # Activate these lines to plot results just from the internal field
            plt.plot(theoretical_ep_sigma_field[-1, :, :][:100, com], 
                    theoretical_ep_sigma_field[-1, :, :][:100, com], 
                    color = "blue", label = "Expected", marker='x')                
            plt.scatter(theoretical_ep_sigma_field[-1, :, :][:100, com], 
                        elastic_nn_sigma_field[-1, :, :][:100, com], 
                        color = "red", label = "ElasticNN", marker='o', 
                        alpha = 0.3)                
            plt.scatter(theoretical_ep_sigma_field[-1, :, :][:100, com], 
                        theoretical_elastic_sigma_field[-1, :, :][:100, com],
                        color = "purple", label = "ElasticTheoretical",
                        marker='+')   
            plt.scatter(theoretical_ep_sigma_field[-1, :, :][:100, com],
                        ep_nn_sigma_field[-1, :, :][:100, com], 
                        color = "green", label = "Calculated", marker='o', 
                        alpha=(0.3))  
            plt.xlabel(r'Expected stress $(Pa)$')
            plt.ylabel(r'Calculated stress $(Pa)$')
            # fig.suptitle("Sigma_" + component[com], fontsize=20)
            # plt.legend()
            # plt.show()
            fig.savefig("ComparisonResults/WithoutBoundaryNodes" + "/sigma_" + 
                        component[com] + ".png", bbox_inches='tight')
            plt.close(fig)
            
            
# for com in range(len(component)):            
#     # if (com in (0, 1, 3)):
#         for point in range(theoretical_ep_epsilon_field.shape[1]):
#             if (com in dum_ind):
#                 fig = plt.figure(figsize=(15, 10))
#                 # if (ML_MODEL_IS_3X3):
#                 plt.plot(range(len(theoretical_ep_epsilon_field[:, point, 
#                          com])), theoretical_ep_epsilon_field[:, point, com], 
#                          color = "blue", label = "Expected", marker='x')                
#                 plt.plot(range(len(theoretical_ep_epsilon_field[:, point, 
#                          com])), elastic_nn_epsilon_field[:, point, com], 
#                          color = "green", label = "ElasticNN", marker='*')    

#                 plt.plot(range(len(theoretical_ep_epsilon_field[:, point, 
#                          com])),  theoretical_elastic_epsilon_field[:, point, 
#                          com], color = "red", label = "ElasticTheoretical", 
#                          marker='o', alpha=(0.3))   
#                 plt.plot(range(len(theoretical_ep_epsilon_field[:, point, 
#                          com])), ep_nn_epsilon_field[:, point, com], 
#                                     color = "purple", label = "Calculated", 
#                                     marker='+', alpha=(0.3))              
#                 plt.xlabel(r'Expected strain $(m/m)$')
#                 plt.ylabel(r'Calculated strain $(m/m)$')
#                 # fig.suptitle("Evolution of Epsilon_" + component[com] + 
#                 # " Material point No. " + str(point), fontsize=20)
#                 # plt.legend()
#                 # plt.show()
#                 fig.savefig( "./ComparisonResults/epsilon_history/" + 
#                 "/Evolution_epsilon_" + component[com] + "point_" + 
#                 str(point) + ".png", bbox_inches='tight')
#                 plt.close(fig)
                
                
# for com in range(len(component)):            
#     # if (com in (0, 1, 3)):
#         for point in range(theoretical_ep_epsilon_field.shape[1]):
#             if (com in dum_ind):
#                 fig = plt.figure(figsize=(15, 10))
#                 # if (ML_MODEL_IS_3X3):
#                 plt.plot(range(len(theoretical_ep_sigma_field[:, point, com])), 
#                          theoretical_ep_sigma_field[:, point, com], 
#                          color = "blue", label = "Expected", marker='x')                
#                 plt.plot(range(len(theoretical_ep_sigma_field[:, point, com])), 
#                          elastic_nn_sigma_field[:, point, com], 
#                          color = "green", label = "ElasticNN", marker='*')    

#                 plt.plot(range(len(theoretical_ep_sigma_field[:, point, com])), 
#                          theoretical_elastic_sigma_field[:, point, com], 
#                          color = "red", label = "ElasticTheoretical", 
#                          marker='o', alpha=(0.3))   
#                 plt.plot(range(len(theoretical_ep_sigma_field[:, point, com])), 
#                          ep_nn_sigma_field[:, point, com], color = "purple", 
#                          label = "Calculated", marker='+', alpha=(0.3))               
#                 plt.xlabel(r'Expected stress $(Pa)$')
#                 plt.ylabel(r'Calculated stress $(Pa)$')
#                 # fig.suptitle("Evolution of Sigma_" + component[com] + 
#                 # " Material point No. " + str(point), fontsize=20)
#                 # plt.legend()
#                 # plt.show()
#                 fig.savefig( "./ComparisonResults/sigma_history/" + 
#                 "/Evolution_sigma_" + component[com] + "point_" + str(point) + 
#                 ".png", bbox_inches='tight')
#                 plt.close(fig)
                
                
                
D_expected = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "D")
D_ep_nn = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                          "D")

D_elastic_expected = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                    str(int(load_inc)) + "/", "D")

D_elastic_nn = deserialise(ROUTE_ELASTIC_NN_MODEL + str(int(load_inc)) + "/", 
                           "D")


D_expected_right = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "D_Right")
D_ep_nn_right = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                          "D_Right")

D_elastic_expected_right = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                    str(int(load_inc)) + "/", "D_Right")

D_elastic_nn_right = deserialise(ROUTE_ELASTIC_NN_MODEL + str(int(load_inc)) + 
                                 "/", "D_Right")


D_expected_down = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "D_Down")
D_ep_nn_down = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                          "D_Down")

D_elastic_expected_down = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                    str(int(load_inc)) + "/", "D_Down")

D_elastic_nn_down = deserialise(ROUTE_ELASTIC_NN_MODEL + str(int(load_inc)) + 
                                 "/", "D_Down")


D_expected_up = deserialise(ROUTE_THEORETICAL_MODEL +  str(int(load_inc)) + 
                                          "/", "D_Up")
D_ep_nn_up = deserialise(ROUTE_EP_NN_MODEL + str(int(load_inc)) + "/", 
                          "D_Up")

D_elastic_expected_up = deserialise(ROUTE_THEORETICAL_ELASTIC_MODEL + 
                                    str(int(load_inc)) + "/", "D_Up")

D_elastic_nn_up = deserialise(ROUTE_ELASTIC_NN_MODEL + str(int(load_inc)) + 
                                 "/", "D_Up")


# # Displacement results are only availabe for the internal field
# for com in range(2):
#     fig = plt.figure(figsize=(15, 10))
#     # if (ML_MODEL_IS_3X3):
#     plt.plot(D_expected[:, com], D_expected[:, com], color = "blue", 
#              label = "Expected", marker='x')                
#     plt.scatter(D_expected[:, com], D_elastic_nn[:, com], color = "red", 
#                 label = "ElasticNN", marker='*', alpha = 0.3)                
#     plt.scatter(D_expected[:, com], D_elastic_expected[:, com], color = "purple", 
#                 label = "ElasticTheoretical", marker='+')   
#     plt.scatter(D_expected[:, com], D_ep_nn[:, com], color = "green", 
#                 label = "Calculated", marker='o', alpha=(0.3))              
#     plt.xlabel(r'Expected displacement $(m)$')
#     plt.ylabel(r'Calculated displacement $(m)$')
#     # fig.suptitle("Epsilon_" + component[com], fontsize=20)
#     # plt.legend()
#     # plt.show()
#     if (com == 0):
#         fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/D_x" + ".png", 
#                     bbox_inches='tight')
#     else:
#         fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/D_y" + ".png", 
#                     bbox_inches='tight')
#     plt.close(fig)
                

D_expected_to_plot = np.concatenate((D_expected, D_expected_right, 
                                     D_expected_down, D_expected_up), axis = 0)
D_ep_nn_to_plot = np.concatenate((D_ep_nn, D_ep_nn_right, D_ep_nn_down, 
                                  D_expected_up), axis = 0)
D_elastic_expected_to_plot = np.concatenate((D_elastic_expected, 
                                             D_elastic_expected_right, 
                                             D_elastic_expected_down, 
                                             D_elastic_expected_up), axis = 0)   
D_elastic_nn_to_plot = np.concatenate((D_elastic_nn, D_elastic_nn_right, 
                                       D_elastic_nn_down, D_elastic_nn_up), 
                                      axis = 0)            
            
# Displacement results are only availabe for the internal field
for com in range(2):
    fig = plt.figure(figsize=(15, 10))
    # if (ML_MODEL_IS_3X3):
    plt.plot(D_expected_to_plot[:, com], D_expected_to_plot[:, com], 
             color = "blue", label = "Expected", marker='x')                
    plt.scatter(D_expected_to_plot[:, com], D_elastic_nn_to_plot[:, com], 
                color = "red", label = "ElasticNN", marker='o', alpha = 0.3)                
    plt.scatter(D_expected_to_plot[:, com], D_elastic_expected_to_plot[:, com], 
                color = "purple", label = "ElasticTheoretical", marker='+')   
    plt.scatter(D_expected_to_plot[:, com], D_ep_nn_to_plot[:, com], 
                color = "green", label = "Calculated", marker='o', alpha=(0.3))              
    plt.xlabel(r'Expected displacement $(m)$')
    plt.ylabel(r'Calculated displacement $(m)$')
    if (com == 0):
        fig.savefig("ComparisonResults/WithBoundaryNodes" +"/D_x" + ".png", 
                    bbox_inches='tight')
    else:
        fig.savefig("ComparisonResults/WithBoundaryNodes" +"/D_y" + ".png", 
                    bbox_inches='tight')
    plt.close(fig)
    
    fig = plt.figure(figsize=(15, 10))
    # if (ML_MODEL_IS_3X3):
    plt.plot(D_expected[:, com], D_expected[:, com], 
             color = "blue", label = "Expected", marker='x')                
    plt.scatter(D_expected[:, com], D_elastic_nn[:, com], 
                color = "red", label = "ElasticNN", marker='o', alpha = 0.3)                
    plt.scatter(D_expected[:, com], D_elastic_expected[:, com], 
                color = "purple", label = "ElasticTheoretical", marker='+')   
    plt.scatter(D_expected[:, com], D_ep_nn[:, com], 
                color = "green", label = "Calculated", marker='o', alpha=(0.3)) 
    plt.xlabel(r'Expected displacement $(m)$')
    plt.ylabel(r'Calculated displacement $(m)$')
    if (com == 0):
        fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/D_x" + ".png", 
                    bbox_inches='tight')
    else:
        fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/D_y" + ".png", 
                    bbox_inches='tight')
    plt.close(fig)
    
    
    # # fig.suptitle("Epsilon_" + component[com], fontsize=20)
    # # plt.legend()
    # # plt.show()
    # if (com == 0):
    #     fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/D_x" + ".png", 
    #                 bbox_inches='tight')
    # else:
    #     fig.savefig("ComparisonResults/WithoutBoundaryNodes" +"/D_y" + ".png", 
    #                 bbox_inches='tight')
    # plt.close(fig)
                
            
            
            
            
            
            


