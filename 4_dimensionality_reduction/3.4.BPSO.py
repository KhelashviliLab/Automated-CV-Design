##### This python code implements BPSO using the NiaPy library. 
##### The code template was obtained from the tutorial provided in the NiaPy website (https://niapy.org/en/stable/tutorials/feature_selection.html).

##### Author: MO (Updated: May 28, 2024) 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=-1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=-1)
        accuracy = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=3, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

df = pd.read_csv('sample_CA_post_variance.csv') # Use the output file obtained after feature selection
dfClassVec = df['class']
df = df.drop('class', axis=1)
feature_names = df.columns.values

X = df.values
y = dfClassVec['class'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

problem = SVMFeatureSelection(X_train, y_train)
task = Task(problem, max_iters=100)
algorithm = ParticleSwarmOptimization(population_size=30, seed=1234)

selected_features = best_features > 0.5
print('Number of selected features:', selected_features.sum())
print('Selected features:', ', '.join(feature_names[selected_features]))

model_selected = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=1)
model_all = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=10, max_samples=1.0/n_estimators), n_jobs=1)

model_selected.fit(X_train[:, selected_features], y_train)
print('Subset accuracy on test:', model_selected.score(X_test[:, selected_features], y_test))
accuracy_cv = cross_val_score(model_selected, X_test[:, selected_features], y_test, cv=5, n_jobs=1).mean()
print('Subset accuracy on test cv:', accuracy_cv)

model_all.fit(X_train, y_train)
print('All Features Accuracy on test:', model_all.score(X_test, y_test))

print(feature_names[selected_features])
dfRED = df[feature_names[selected_features]]
dfRED.to_pickle(path='sample_CA_post_variance_bpso_result.zip', compression='zip') # Save the output file as zip but can be changed to save it as csv
