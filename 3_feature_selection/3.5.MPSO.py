##### This python code implements MPSO using the NiaPy library. 
##### The code template was first obtained from the tutorial provided in the NiaPy website (https://niapy.org/en/stable/tutorials/feature_selection.html) and then modified to perform MPSO as described in our paper.
##### Note that once you obtain the final reduced feature set, you have to extract it from the original dataset manually (e.g., using Jupyter notebook) or modify the code to save the final reduced feature set as MPSO.csv.

##### Author: HX (Updated: May 30, 2024) 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# User Inputs
Total_Dimensions = 5
constrain_aim_rescount = 3
constrain_aim_respair = 1
Preselected_features = "".split(", ")
input_feature = 'sample_CA_post_variance.csv' # Use the output file obtained after feature selection
iteration = 1 # Change the iteration number after each run ends

if iteration == 1:
    selected_in_previous_step = 1 
if iteration > 1:
    selected_in_previous_step = np.load("./selected_feature_matrix_iter%d.npy"%(iteration-1)).astype(bool)

feature_excluding_ratio = 0.8
n_estimators=10

start_time = time.time()

class SVMFeatureExpand(Problem):
    def __init__(self, Available_data_map,Preselected_data_map, X_train, y_train, alpha=0.99):
        super().__init__(dimension=Available_data_map.shape[0]*Available_data_map.shape[1], lower=0, upper=1)
        self.Available_data_map = Available_data_map
        self.Preselected_data_map = Preselected_data_map
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
    def _evaluate(self, x):
        selected = x > feature_excluding_ratio
        selected_matrix_temp = selected.reshape(self.Available_data_map.shape[0],self.Available_data_map.shape[1])
        selected_matrix = (selected_matrix_temp*self.Available_data_map + self.Preselected_data_map).astype(bool)
        num_selected = np.sum(selected_matrix,axis=0)
        n_selected_residues = np.zeros(self.Available_data_map.shape[1])
        for dim in range(self.Available_data_map.shape[1]):
            residue_in_dim = np.concatenate((feature_index1[selected_matrix[:,dim]],feature_index2[selected_matrix[:,dim]]))
            n_selected_residues[dim] = len(np.unique(residue_in_dim))
        sum_selected_data = np.matmul(self.X_train,selected_matrix)
        data_averaged = np.divide(sum_selected_data, num_selected, out=np.zeros_like(sum_selected_data), where=num_selected!=0)
        if np.sum(num_selected)==0:
            return 1.0
        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=-1, n_estimators=n_estimators, max_samples=1.0/n_estimators), n_jobs=-1)
        accuracy = cross_val_score(clf, data_averaged, self.y_train, cv=3, n_jobs=-1).mean()
        score = 1 - accuracy
        n_selected_residues = (n_selected_residues-constrain_aim_respair).clip(min=0)
        #objfunc = self.alpha * score + (1 - self.alpha) * (np.sum(np.divide(n_selected_residues, n_residues, out=np.zeros_like(n_selected_residues), where=n_residues!=0)))
        num_selected_forobj = (num_selected-constrain_aim_rescount).clip(min=0)
        objfunc = self.alpha * score + (1 - self.alpha) * (np.sum(np.divide(num_selected_forobj, num_features, out=np.zeros_like(num_selected_forobj), where=num_features!=0, casting='unsafe'))+np.sum(np.divide(n_selected_residues, n_residues, out=np.zeros_like(n_selected_residues), where=n_residues!=0))) 
        print("Acc:", accuracy)
        print("Obj:", objfunc)
        return objfunc

df = pd.read_csv(input_feature, compression='infer')
y = df['class'].to_numpy()
df.drop(columns=['class'],inplace=True)

feature_names = df.columns.values
residue_list = np.sort(np.unique(','.join([','.join(i.split("res")[1].split(".")) for i in feature_names]).split(",")).astype(int))
feature_index1 = np.array([np.where(residue_list==int(i.split("res")[1].split(".")[0]))[0][0] for i in feature_names])
feature_index2 = np.array([np.where(residue_list==int(i.split("res")[1].split(".")[1]))[0][0] for i in feature_names])

Preselected_index = [np.where(feature_names==n)[0][0] for n in Preselected_features if n!=""]
Available_data_map = np.zeros((df.shape[1],Total_Dimensions)).astype(bool)
Available_data_map = (Available_data_map + selected_in_previous_step).astype(bool)
num_features = np.sum(Available_data_map,axis=0)
n_residues = np.zeros(Total_Dimensions)
for dim in range(Total_Dimensions):
    residue_in_dim = np.concatenate((feature_index1[Available_data_map[:,dim]],feature_index2[Available_data_map[:,dim]]))
    n_residues[dim] = len(np.unique(residue_in_dim))

Preselected_data_map = np.zeros((df.shape[1],Total_Dimensions)).astype(bool)
for i in range(len(Preselected_index)):
    Preselected_data_map[Preselected_index[i],i] = 1

print("Preselected features:")
print(feature_names[np.where(np.sum(Preselected_data_map,axis=1))[0]])
print("Avaliable features in each dimension:")
print(num_features)
print("Related residues in each dimension:")
print(n_residues)
print("------------------------")

X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

problem = SVMFeatureExpand(Available_data_map,Preselected_data_map,X_train, y_train)
task = Task(problem, max_iters=200)
algorithm = ParticleSwarmOptimization(population_size=30, seed=14)
# population_size=25, c1=2.0, c2=2.0, w=0.7, min_velocity=-1.5, max_velocity=1.5, repair=<function reflect>, *args, **kwargs
best_features, best_fitness = algorithm.run(task)

selected_matrix = (((best_features > feature_excluding_ratio).reshape(Available_data_map.shape[0],Available_data_map.shape[1]))*Available_data_map + Preselected_data_map).astype(bool)
np.save('./selected_feature_matrix_iter%d.npy'%iteration,selected_matrix)
num_selected = np.sum(selected_matrix,axis=0)
n_selected_residues = np.zeros(Total_Dimensions)
for dim in range(Total_Dimensions):
    residue_in_dim = np.concatenate((feature_index1[selected_matrix[:,dim]],feature_index2[selected_matrix[:,dim]]))
    n_selected_residues[dim] = len(np.unique(residue_in_dim))

print('Number of selected features:', num_selected)
print('Number of related residues:', n_selected_residues)
if np.max(num_selected)<20:
    for i in range(Total_Dimensions):
        print('Selected features in Dim%d:' %i, ', '.join(feature_names[selected_matrix[:,i]]))

model_selected = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=1)
model_all = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=1)

sum_selected_data_train = np.matmul(X_train,selected_matrix)
data_averaged_train = np.divide(sum_selected_data_train, num_selected, out=np.zeros_like(sum_selected_data_train), where=num_selected!=0)

sum_selected_data_test = np.matmul(X_test,selected_matrix)
data_averaged_test = np.divide(sum_selected_data_test, num_selected, out=np.zeros_like(sum_selected_data_test), where=num_selected!=0)

model_selected.fit(data_averaged_train, y_train)
print('Subset accuracy:', model_selected.score(data_averaged_test, y_test))
print('Subset CV accuracy:', cross_val_score(model_selected, data_averaged_test, y_test, cv=5, n_jobs=1).mean())

if len(Preselected_index)>0:
    model_all.fit(X_train[:,Preselected_index], y_train)
    print('chosen Features Accuracy:', model_all.score(X_test[:,Preselected_index], y_test))
    print('chosen Features CV Accuracy:', cross_val_score(model_all, X_test[:,Preselected_index], y_test,cv=5, n_jobs=1).mean())


