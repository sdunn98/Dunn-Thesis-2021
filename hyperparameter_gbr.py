import time
from math import log
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
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
def gbr_model(X_data, Y_data, lr, n):

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(learning_rate=lr, n_estimators=n)
    model.fit(X_train, Y_train)

    # Prediction
    prediction = model.predict(X_test)
    R2 = metrics.r2_score(Y_test, prediction)

    return R2

# ----------------------- GRID SEARCH ----------------------- #
# a_list = np.linspace(0,0.5,20)
# b_list = np.linspace(10,1000,20)

a_list = [0.1,0.2]
b_list = [800,900,1000]

search_points = len(a_list)*len(b_list)

R_max = 0
C_opt = 0
G_opt = 0

count = 0
for i in a_list:
    for j in b_list:
        R = gbr_model(X_data=X, Y_data=Y, lr=i, n=j)

        if R > R_max:
            R_max = R
            C_opt = i
            G_opt = j

        count += 1
        print(f"\rConducting grid search for optimal hyperparameters -> {100*count/search_points:0.2f}%", end='')

print(f"\nLR = {C_opt}, N = {G_opt}")

# ------------------- REFINED GRID SEARCH ------------------- #
# G1 = log(G_opt, 10) - 0.5
# G2 = log(G_opt, 10) + 0.5
# C1 = log(C_opt, 10) - 1
# C2 = log(C_opt, 10) + 1

# a_list = [C_opt-0.06, C_opt-0.02, C_opt-0.01, C_opt, C_opt+0.02, C_opt+0.04, C_opt+0.06]
a_list = [C_opt-0.01, C_opt, C_opt+0.01]
# b_list = [G_opt-50, G_opt-25, G_opt, G_opt+25, G_opt+50]
b_list = [G_opt-25, G_opt, G_opt+25]
search_points = len(a_list)*len(b_list)

Gs = []
Cs = []
Rs = []

count = 0
print(' ')
for i in a_list:
    for j in b_list:
        R = gbr_model(X_data=X, Y_data=Y, lr=i, n=j)
        Gs.append(j)
        Cs.append(i)
        Rs.append(R)

        if R > R_max:
            R_max = R
            C_opt = i
            G_opt = j

        count += 1
        print(f"\rRefining grid search for optimal hyperparameters -> {100*count/search_points:0.2f}%", end='')

print('\nOptimal Parameters:')
print('\tLR = {:.3f}'.format(C_opt))
print('\tN = {:.2f}'.format(G_opt))
print('\tR2 = {:.4f}'.format(R_max))

# ---------------------- CONTOUR PLOT ---------------------- #
fig = plt.figure(figsize=(7,7), dpi=300)
contour = plt.tricontourf(Cs, Gs, Rs, cmap="YlGnBu", levels=24)
fig.colorbar(contour)


plt.title('GBR Hyperparameter Optimisation', fontsize='large')
plt.xlabel('Learning Rate Parameter Value', fontsize='medium')
plt.ylabel('N Estimators Parameter Value', fontsize='medium')
plt.xscale('linear')
plt.yscale('linear')
plt.annotate('$R^2$ = {:.4f}'.format(R_max), xy=(C_opt*1.1, G_opt*1.03), color='white',
             fontsize='medium', style='italic')
plt.annotate('({:.2f}, {})'.format(C_opt, G_opt), xy=(C_opt*1.1, G_opt*0.95), color='white',
             fontsize='small', style='italic')

plt.plot(C_opt, G_opt, marker='o', color='white', markersize=4)

savename = ''
words = label.split()
for word in words:
    savename += word[0].upper()

savename += "-GBR"

if saveimg:
    plt.savefig('figures/hyperparameters/' + savename + '.png')
    print(f"Hyperparameter plot saved to... figures/hyperparameters/{savename}.png")

# plt.show()

end = time.time()
print('Time Elapsed = {:.1f} seconds'.format(end-start))