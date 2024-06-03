#import amino as amino
import amino_fast_mod as amino
import numpy as np
import matplotlib.pyplot as plt

### STEP 1. Read data
colvar = open("data/COLVAR_411_descriptors.csv")  #colvar = open("data/BOUND_COLVAR")
split = colvar.readline().split(",")    #split = colvar.readline().split()
split[-1] = split[-1][:-1]

names = []
trajs = {}

# OP (feature) names
for i in range(1, len(split)):
    names.append(split[i])
    trajs[split[i]] = []
# OP (feature) values
for line in colvar:
    timestep = line.split(",")          #timestep = line.split()
    timestep[-1] = timestep[-1][:-1]
    for i in range(len(timestep) - 1):
        trajs[names[i]].append(float(timestep[i + 1]))

all_ops = []

for i in names:
    all_ops.append(amino.OrderParameter(i, trajs[i]))

print("We have %d OPs in total!" % len(all_ops))
print("OPs: %s" % names)

### STEP 2. Perform AMINO
final_ops = amino.find_ops(all_ops, 10, 20, distortion_filename='distortion_array')   # find_ops(old_ops, max_outputs=20, bins=20, bandwidth=None, kernel='epanechnikov', distortion_filename=None, return_memo=False, weights=None)

myfile2 = open('output2_final_OPs_n10_p20.txt', 'w')
print("\nAMINO order parameters:")
for i in final_ops:
    myfile2.write("%s\n" % i)
    print(i)
myfile2.close()

myfile3 = open('output3_distortion_array_n10_p20.txt', 'w')
data_array = np.load('distortion_array.npy')
myfile3.write("%s\n" % data_array)
print("\nData summary:\n", data_array)
myfile3.close()

### STEP 3. Draw distortion graphs
