

import os
import bioread
import numpy as np
import pickle
from Person import Person
from functions_for_project import read_biopac_file



# Folder containing the BIOPAC files
folder_path = "C:\\Users\\nived\\OneDrive - Afeka College Of Engineering\\Medical Engineering folder\\Final Project\\Data&Files\\code"

# List of names for the data matrices
names = ["orit", "simon", "laurence", "emma", "niv", "jordan", "yuval"]
names.sort()

# List to store Person objects
persons = []
a = os.listdir(folder_path)
# Iterate over each file in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".acq") and i <= len(names):
        file_path = os.path.join(folder_path, filename)
        # Read BIOPAC file and create data matrix
        data_matrix = read_biopac_file(file_path)[:, 0:3]  # Assuming you only want the first 3 columns
        time_vec = bioread.read_file(file_path).time_index
        time_vec = np.around(time_vec,3)
        # Create Person object and add to persons list
        person = Person(names[i], data_matrix,time_vec)
        persons.append(person)
        # Create variable for Person object
        variable_name = f"p_{names[i]}"
        globals()[variable_name] = person
        print(f"Created Person object for {names[i]} with variable name: {variable_name}")

p_orit.timestamps_list = [0,177.470,374.770,530.515,683.005,868.600,1067.185,1281.445,1481.645,1678.725]
p_yuval.timestamps_list = [0,206.535,574.735,1087.410,1655.505,2138.630,2557.915,3009.400,3387.780,3724.805]
p_simon.timestamps_list = [0,177.965,375.375,530.625,680.311,864.400,1063.935,1278.030,1478.335,1675.100]
p_emma.timestamps_list = [[7.140,133.610,254.180,358.230,459.870,612.100,746.395,865.380,1025.065,1163.175],
                          [1318.195,1404.630,1501.880,1611.350,1741.195,1853.775,1970.620,2079.765,2206.145,2321.560]]
p_niv.timestamps_list = [[0,86.525,170.555,256.980,347.135,419.690,500.925,603.210,704.925,817.655],
                         [938.465,1028.830,1113.780,1204.245,1289.180,1384.805,1489.780,1584.955,1692.615,1793.780]]
p_jordan.timestamps_list = [0,243.141,505.540,699.645,977.375,1300.065,1590.470,1829.795,1996.895,2211.085]
p_laurence.timestamps_list = [0,224.190,502.955,777.485,1030.945,1281.155,1564.260,1863.820,2206.185,2529.505]
p_yuval.timestamps_list = [0,206.535,574.735,1087.410,1655.505,2138.630,2557.915,3009.400,3387.780,3724.805]

# Dividing the measurements into sections for each person


for p in persons:
    if len(p.timestamps_list) == 10:
        keys = list(p.measurements.keys())
        matrix = p.data_matrix
        timestamps = p.timestamps_list
        time_vec = p.time
        index = []
        for i in range(len(keys)):
            if len(timestamps) == 10:
                #print(i)
                #index = (np.where(time_vec==timestamps[i])[0],np.where(time_vec==timestamps[i+1])[0])
                index.append(int(np.where(time_vec==timestamps[i])[0]))
        #print(index)
        for i in range(len(keys)):
            if i < len(keys)-1:
                p.measurements[keys[i]] = matrix[index[i]:index[i+1]-1, :]
            else:
                p.measurements[keys[i]] = matrix[index[i]:, :]
        print(p.name)
    elif len(p.timestamps_list) == 2:
        keys = list(p.measurements.keys())
        matrix = p.data_matrix
        timestamps = p.timestamps_list
        time_vec = p.time
        #print(p.name)
        if len(timestamps[0]) == 10 and len(timestamps[1]) == 10:
            index = [[],[]]
            for i in range(len(keys)):
                #print(i)
                index[0].append(int(np.where(time_vec==timestamps[0][i])[0]))
                index[1].append(int(np.where(time_vec==timestamps[1][i])[0]))
            #print(index)
            for i in range(len(keys)):
                if i < len(keys)-1:
                    concatenated_matrix = np.concatenate((matrix[index[0][i]:index[0][i+1], :], matrix[index[1][i]:index[1][i+1], :]), axis=0)
                else:
                    concatenated_matrix = np.concatenate((matrix[index[0][i]:, :], matrix[index[1][i]:, :]), axis=0)
                p.measurements[keys[i]] = concatenated_matrix
            print(p.name)
        else:
            print("Timestamps lists do not have lengths equal to 10.")

file_path = "persons_data.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(persons, file)
    print(f"Created .pkl file with the name: '{file_path}'")


