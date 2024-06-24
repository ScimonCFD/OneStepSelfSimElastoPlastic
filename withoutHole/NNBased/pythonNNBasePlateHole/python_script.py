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

import pickle

# def deserialise_flag():
#     with open('flag.pkl', 'rb') as f:
#         flag = pickle.load(f)
#     f.close()
#     return flag

def serialise_fields():
    np.save(str(int(time))+'/epsilon', epsilon)
    # np.save(str(int(time))+'/epsilonExpected', epsilonExpected) 
    np.save(str(int(time))+'/sigma', sigma)   
    # np.save(str(int(time))+'/DExpected', DExpected) 
    # np.save(str(int(time))+'/sigmaExpected', sigmaExpected)   
    np.save(str(int(time))+'/D', D)   

    print("\n \n \n \n time is ", time)
    # print(states_1)
    # for i in range(int(nStates)):
    #     np.save(str(int(time))+'/states_' + str(i), '/states_' + str(i))

