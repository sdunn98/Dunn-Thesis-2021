import time
from math import log
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split

start = time.time()

# ---------------------- READ DATASETS ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
importances = 'datasets_processed/feature-importance.csv'

df = pd.read_csv(dataset)
importance_df = pd.read_csv(importances)

saveimg = True

target = 'Vacancy energy [eV/O atom]'                # 'Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'
label = 'Vacancy Energy'                                 # 'Formation Energy', 'Vacancy Energy', 'Band Gap'

# ----------------------- PREPARE DATA ----------------------- #
features_FE = list(importance_df['Formation energy [eV/atom]'])
features_VE = list(importance_df['Vacancy energy [eV/O atom]'])
features_BG = list(importance_df['Band gap [eV]'])

if target == 'Formation energy [eV/atom]':
    features = features_FE
elif target == 'Vacancy energy [eV/O atom]':
    features = features_VE
elif target == 'Band gap [eV]':
    features = features_BG
else:
    features = list(df.columns)

# Feature Importance
todrop = []
for i in df.columns:
    if i not in features:
        todrop.append(i)

# Feature Reduction
X = df.drop(todrop, axis='columns')
Y = df[target]

# ---------------- SUPPORT VECTOR REGRESSION ---------------- #
def svm_model(X_data, Y_data, c, g):
    #
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=False)
    model = NuSVR(C=c, gamma=g)
    model.fit(X_train, Y_train)

    # Prediction
    prediction = model.predict(X_test)
    R2 = metrics.r2_score(Y_test, prediction)

    return R2

# ----------------------- GRID SEARCH ----------------------- #
gamma_list = np.logspace(-3, 2, 10)
c_list = np.logspace(-3, 2, 10)
search_points = len(gamma_list)*len(c_list)

R_max = 0
C_opt = 0
G_opt = 0

count = 0
for i in gamma_list:
    for j in c_list:
        R = svm_model(X_data=X, Y_data=Y, c=j, g=i)

        if R > R_max:
            R_max = R
            C_opt = j
            G_opt = i

        count += 1
        print(f"\rConducting grid search for optimal SVR hyperparameters -> {100*count/search_points:0.2f}%", end='')

print(f"\nC = {C_opt}, gamma = {G_opt}")

# ------------------- REFINED GRID SEARCH ------------------- #
G1 = log(G_opt, 10) - 0.6
G2 = log(G_opt, 10) + 0.6
C1 = log(C_opt, 10) - 1
C2 = log(C_opt, 10) + 1

gamma_list = np.logspace(G1, G2, 20)
c_list = np.logspace(C1, C2, 20)
search_points = len(gamma_list)*len(c_list)

Gs = []
Cs = []
Rs = []

count = 0
print(' ')
for i in gamma_list:
    for j in c_list:
        R = svm_model(X_data=X, Y_data=Y, c=j, g=i)
        Gs.append(i)
        Cs.append(j)
        Rs.append(R)

        if R > R_max:
            R_max = R
            C_opt = j
            G_opt = i

        count += 1
        print(f"\rRefining grid search for optimal SVR hyperparameters -> {100*count/search_points:0.2f}%", end='')

print('\nOptimal Parameters:')
print('\tC = {:.3f}'.format(C_opt))
print('\tGamma = {:.2f}'.format(G_opt))
print('\tR2 = {:.3f}'.format(R_max))

# ---------------------- CONTOUR PLOT ---------------------- #
fig = plt.figure(figsize=(8,7), dpi=180)
contour = plt.tricontourf(Cs, Gs, Rs, cmap="YlGnBu", levels=24)
fig.colorbar(contour)

plt.xscale('log')
plt.yscale('log')
plt.title(label + ' - SVM Hyperparameter Optimisation', fontsize='large')
plt.xlabel('C Parameter Value', fontsize='medium')
plt.ylabel('Gamma Parameter Value', fontsize='medium')

plt.annotate('$R^2$ = {:.4f}'.format(R_max), xy=(C_opt*1.1, G_opt*1.03), color='white',
             fontsize='medium', style='italic')
plt.annotate('({:.2f}, {:.3f})'.format(C_opt, G_opt), xy=(C_opt*1.1, G_opt*0.95), color='white',
             fontsize='small', style='italic')

plt.plot(C_opt, G_opt, marker='o', color='white', markersize=4)

savename = "NuSVR-hyperparameter-"
words = label.split()
for word in words:
    savename += word[0].upper()

if saveimg:
    plt.savefig('figures/hyperparameters/' + savename + '.png')
    print(f"Hyperparameter plot saved to... figures/hyperparameters/{savename}.png")

# plt.show()

end = time.time()
print('Time Elapsed = {:.1f} seconds'.format(end-start))