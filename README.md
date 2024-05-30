# Automated Collective Variable Discovery 
Myongin Oh, Margarida Rosa, Hengyi Xie, George Khelashvili at Weill Cornell Medicine. 
Please refer to the paper: https://www.biorxiv.org/content/10.1101/2024.04.19.590308v1 

## Author Information
- **Authors**: MO, MR, HX (Latest Update: May 28, 2024) 
- **Contact Information**: MO (myo4001@med.cornell.edu) MR (mar4026@med.cornell.edu) GK (gek2009@med.cornell.edu) 
If you have any questions or notice any mistakes, please contact us.

## Description
Our protocol addresses the challenge of sampling transitions between long-lived metastable states in biomolecules by automating the detection of collective variables (CVs) for enhanced sampling in molecular dynamics (MD) simulations. Traditional CV selection relies heavily on intuition and prior knowledge, which can introduce bias and limit mechanistic insights. We circumvent this by utilizing machine learning algorithms to identify CVs automatically.

## Usage
This protocol is divided into four steps/directories: Please refer to the README file in each directory for more detailed information about each step.

1. **Input MD Data**: Contains all the inputs required for the subsequent steps.

2. **Feature Extraction**: Includes scripts for extracting Cartesian coordinates from molecular dynamics trajectories, merging CSV files using pandas, and trimming the dataset based on variance.

3. **Feature Selection**: Contains scripts for different feature calculation and selection methods (Chi-square-AMINO, Fisher-AMINO, BPSO, and MPSO)

4. **Dimensionality Reduction**: Includes scripts for different dimensionality reduction methods (PCA, FLDA, MHLDA, ZHLDA, GDHLDA).

Please refer to the following worflow:

![image](https://github.com/KhelashviliLab/Automated-CV-Design/assets/99993156/5d7eeaf7-aedb-42a9-b2a6-a34d79b34732)
