import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split


# ---------------------- READ DATASETS ---------------------- #
df = pd.read_csv('datasets_processed/db-norm-shuffled.csv')
atoms_df = pd.read_csv('datasets_processed/db-atoms.csv').set_index('symbol')
scalers_df = pd.read_csv('datasets_processed/norm-scalers.csv').set_index('Feature')
importance_df = pd.read_csv('datasets_processed/feature-importance.csv')

# ---------------------- CLEAN DATASET ---------------------- #
def clean(dataset):
    dataset = dataset.replace('-', np.NaN)
    dataset = dataset.dropna()
    dataset = dataset.apply(pd.to_numeric, errors='ignore')
    return dataset

# ------------------------ NORMALISE ------------------------ #
def normalise(val, feature):
    minn = scalers_df.loc[feature].Min
    maxx = scalers_df.loc[feature].Max
    norm_val = (val - minn)/(maxx - minn)
    return norm_val

# ---------------------- UN-NORMALISE ---------------------- #
def unnormalise(norm_val, feature):
    minn = scalers_df.loc[feature].Min
    maxx = scalers_df.loc[feature].Max
    val = norm_val * (maxx - minn) + minn
    return val

# -------------------- DATA PREPARATION -------------------- #
features_FE = list(importance_df['Formation energy [eV/atom]'])[-10:]
features_VE = list(importance_df['Vacancy energy [eV/O atom]'])[-10:]
todrop = []

for i in range(len(atoms_df.columns)):
    if atoms_df.count()[i] < 70 and atoms_df.columns[i] not in features_FE and atoms_df.columns[i] not in features_VE:
        todrop.append(atoms_df.columns[i])

delete = ['H','Be','Cs','Fr','Ra','Rf','Db','Sg','Tc','Bh','Hs','Mt','Ds','Rg','Cd','Cn','Tl','Nh','Pb','Fl','As',
          'Mc','Se','Po','Lv','At','Ts','He','Ne','Ar','Kr','Xe','Rn','Og','Pm','Eu','Ho','Er','Tm','Ac','Th','Pa','U',
          'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']

atoms_df = atoms_df.drop(delete, axis='index')
atoms_df = atoms_df.drop(todrop, axis='columns')
atoms_df = clean(atoms_df)
print(f"Number of input features = {2 * len(atoms_df.columns)}\n")

# ------------------ FORMATION ENERGY MODEL ------------------ #
print("Formation Energy Model:")
X = df.drop(['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'], axis='columns')
X = X.drop(["A " + i.replace("_", " ") for i in todrop], axis='columns')
X = X.drop(["B " + i.replace("_", " ") for i in todrop], axis='columns')
Y_FE = df['Formation energy [eV/atom]']

# FE model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_FE, test_size=0.20, shuffle=False)

svr_FE = NuSVR(nu=0.4, C=23.36, gamma=0.08)
gbr_FE = GradientBoostingRegressor(learning_rate=0.2, n_estimators=500)

model_FE = VotingRegressor(estimators=[('nusvr', svr_FE), ('gbr', gbr_FE)], weights=[0.7, 0.3])
model_FE.fit(X_train, Y_train)

# Prediction
prediction = model_FE.predict(X_test)
R2 = metrics.r2_score(Y_test, prediction)
RMSE = metrics.mean_squared_error(Y_test, prediction)
MAE = metrics.mean_absolute_error(Y_test, prediction)
print(f'R^2 = {R2}')
print(f'RMSE = {RMSE}')
print(f'MAE = {MAE}')


# ------------------ VACANCY ENERGY MODEL ------------------ #
print("\nVacancy Energy Model:")
X = df.drop(['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'], axis='columns')
X = X.drop(["A " + i.replace("_", " ") for i in todrop], axis='columns')
X = X.drop(["B " + i.replace("_", " ") for i in todrop], axis='columns')
Y_VE = df['Vacancy energy [eV/O atom]']

# VE model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_VE, test_size=0.20, shuffle=False)

svr_VE = NuSVR(nu=0.4, C=31.41, gamma=0.179)
gbr_VE = GradientBoostingRegressor(learning_rate=0.25, n_estimators=500)

model_VE = VotingRegressor(estimators=[('nusvr', svr_VE), ('gbr', gbr_VE)], weights=[0.7, 0.3])
model_VE.fit(X_train, Y_train)

# Prediction
prediction = model_VE.predict(X_test)
R2 = metrics.r2_score(Y_test, prediction)
RMSE = metrics.mean_squared_error(Y_test, prediction)
MAE = metrics.mean_absolute_error(Y_test, prediction)
print(f'R^2 = {R2}')
print(f'RMSE = {RMSE}')
print(f'MAE = {MAE}\n')


# --------------------- BAND GAP MODEL --------------------- #
print("\nBand Gap Model:")
X = df.drop(['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'], axis='columns')
X = X.drop(["A " + i.replace("_", " ") for i in todrop], axis='columns')
X = X.drop(["B " + i.replace("_", " ") for i in todrop], axis='columns')

Y_BG = []
for i in df['Band gap [eV]']:
    if i == 0:
        Y_BG.append(0)
    else:
        Y_BG.append(1)

# BG model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_BG, test_size=0.20, shuffle=False)

model_BG = KNeighborsClassifier(n_neighbors=3, p=3)
model_BG.fit(X_train, Y_train)

# Prediction
prediction = model_BG.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, prediction)
precision = metrics.precision_score(Y_test, prediction, average=None)
print(f"Precision score (zero prediction) = {precision[0]:.3f}")
print(f"Accuracy score = {accuracy:.3f}\n")

# ---------------- SOLUTION SEARCH ALGORITHM ---------------- #
print(f"Number of atoms considered = {len(atoms_df)}")
print("Searching for solutions...")

# Normalise atoms data
atoms_norm_df = pd.DataFrame()

for i in atoms_df.columns:
    norm_values = []
    for j in atoms_df[i]:
        norm_values.append(normalise(j, i.replace('_', ' ')))

    atoms_norm_df[i] = norm_values

atoms_norm_df['symbol'] = atoms_df.index
atoms_norm_df.set_index('symbol', inplace=True)

progress = 0
search_points = len(atoms_norm_df)**2
solutions = []
found = 0

# Search for solutions
for i in atoms_norm_df.index:
    for j in atoms_norm_df.index:
        props = []
        # Normalise atom properties
        for k in atoms_norm_df.columns:
            props.append(atoms_norm_df.loc[i][k])
            props.append(atoms_norm_df.loc[j][k])

        # Target property predictions
        FE = model_FE.predict([props])[0]
        VE = model_VE.predict([props])[0]
        BG = model_BG.predict([props])[0]

        # Unnormalise predictions
        FE = unnormalise(FE, 'Formation energy [eV/atom]')
        VE = unnormalise(VE, 'Vacancy energy [eV/O atom]')

        # List solutions
        if FE < -2.6 and VE < -2 and VE > -5 and BG == 0:
            found += 1
            solutions.append([i,j,FE,VE,BG])

        progress += 1
        print(f"\rSolutions found = {found} -> Progress = {100 * progress / search_points:0.2f}%", end='')


for i in solutions:
    print(f"{i[0]}{i[1]} -> FE = {i[2]:0.4f}, VE = {i[3]:0.4f}, BG = {i[4]}")


