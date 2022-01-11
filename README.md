# Dunn-Thesis-2021
Contains the code relating to the honours thesis written by Sebastian Dunn under supervision of Dr. Ruth Knibbe, titled "Application of Machine Learning in New Cathode Material Design".

## Overview
datasets_original contains the original datasets used in this project.
datasets_processed contains the datasets produced after processing and the output results of various files for storage and use in other files.

data_preprocessing.py conducts the preprocessing of the data including cleaning and normalising the datasets, but most significantly combining the two original datasets into one for use throughout the project.
feature_correlation.py outputs a correlation matrix for data analysis.
feature_importance.py outputs the relative feature importance for each target property.
feature_reduction.py assesses the impact of feature reduction on the regressor models.
feature_reduction_bg.py assesses the impact of feature reduction on the band gap classifier model.
hyperparameter_gbr.py contains the hyperparameter optimisation for the GBR model.
hyperparameter_svr.py contains the hyperparameter optimisation for the SVR model.
model_classifier.py contains the final band gap classifier model.
model_vote.py contains the final voting regressor model for the formation energy and vacancy energy as well as comparison to the individual regressors.
solution_search.py searches through all possible atom combinations using the voting regressor and classifier models to find compounds which satisfy the desired material property ranges.
