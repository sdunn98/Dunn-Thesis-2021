import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor

from sklearn import metrics
from sklearn.model_selection import train_test_split


# ---------------------- READ DATASETS ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
importances = 'datasets_processed/feature-importance.csv'

df = pd.read_csv(dataset)
importance_df = pd.read_csv(importances)
scalers_df = pd.read_csv('datasets_processed/norm-scalers.csv').set_index('Feature')

saveimg = True
plotimg = True

# ---------------------- Un-normalise ---------------------- #
def unnormalise(norm_val, feature):
    minn = scalers_df.loc[feature].Min
    maxx = scalers_df.loc[feature].Max
    val = norm_val * (maxx - minn) + minn
    return val

# ------------------- VOTING REGRESSOR ------------------- #
def regressor_model(target: str, label: str, features, model):
    # Features
    todrop = []
    for i in df.columns:
        if i not in features:
            todrop.append(i)

    X = df.drop(todrop, axis='columns')
    Y = df[target]

    # RFR model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

    # Evaluate Model Accuracy
    acronym = ''.join([i for i in str(model).split("(")[0] if i.isupper()])
    if acronym == 'NSVR': acronym = 'NuSVR'

    print(f"Model = {acronym}")

    model.fit(X_train, Y_train)

    # Prediction
    # prediction = model.predict(X_test)
    # R2 = metrics.r2_score(Y_test, prediction)
    # RMSE = metrics.mean_squared_error(Y_test, prediction)
    # MAE = metrics.mean_absolute_error(Y_test, prediction)

    prediction = model.predict(X_train)
    R2 = metrics.r2_score(Y_train, prediction)
    RMSE = metrics.mean_squared_error(Y_train, prediction)
    MAE = metrics.mean_absolute_error(Y_train, prediction)

    print(f'R^2 = {R2}')
    print(f'RMSE = {RMSE}')
    print(f'MAE = {MAE}\n')

    f = lambda x: unnormalise(x, target)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Y_test, prediction = zip(*sorted(zip(Y_test, prediction)))
    # ax.plot(f(np.array(Y_test)), f(np.array(prediction)), marker='o', linestyle='none', markersize=3.5)

    Y_train, prediction = zip(*sorted(zip(Y_train, prediction)))
    ax.plot(f(np.array(Y_train)), f(np.array(prediction)), marker='o', linestyle='none', markersize=2.5)

    a = scalers_df.loc[target].Min
    b = scalers_df.loc[target].Max
    lims = (a, b)

    ax.plot(lims, lims, 'k--', alpha=0.8, zorder=0)

    ax.set_aspect('equal')
    ax.set_xlabel('Actual ' + label)
    ax.set_ylabel('Predicted ' + label)

    ax.annotate('$R^2$ = {:.3f}'.format(R2), xy=(0.1, 0.85), xycoords='axes fraction')
    ax.annotate('RMSE = {:.4f}'.format(RMSE), xy=(0.1, 0.8), xycoords='axes fraction')
    ax.annotate('MAE = {:.4f}'.format(MAE), xy=(0.1, 0.75), xycoords='axes fraction')

    title = acronym if acronym != 'VR' else 'Voting Regressor'
    plt.title(title + ' - ' + 'Training Dataset')
    # plt.title(title + ' - ' + label)
    fig.tight_layout()

    # plt.show()
    if saveimg == True:
        plt.savefig('figures/model_evaluation/' + acronym + '-' + label.lower().replace(' ','-') + '-train', dpi=300)

    return R2


# Choose features to include
targets = ['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]']
features = list(df.columns)
for i in targets:
    features.remove(i)

# Models
def reg1(c, g):
    return NuSVR(nu=0.4, C=c, gamma=g)

reg2 = RandomForestRegressor(n_estimators=500)

def reg3(lr):
    return GradientBoostingRegressor(learning_rate=lr, n_estimators=500)

def gbr(lr):
    return GradientBoostingRegressor(n_estimators=lr)

def vote3(c, g, w, lr):
    return VotingRegressor(estimators=[('nusvr', reg1(c, g)), ('rfr', reg2), ('gbr', reg3(lr))], weights=[w, (1-w)/2, (1-w)/2])

def vote2(c, g, w, lr):
    return VotingRegressor(estimators=[('nusvr', reg1(c, g)), ('gbr', reg3(lr))], weights=[w, 1-w])

# FE Hyperparameters 23.36, 0.08
# VE Hyperparameters 31.41, 0.179

# print("Modeling for Formation Energy")
# regressor_model('Formation energy [eV/atom]', 'Formation Energy', features, reg1(c=23.36, g=0.08))
# regressor_model('Formation energy [eV/atom]', 'Formation Energy', features, reg2(100))
# regressor_model('Formation energy [eV/atom]', 'Formation Energy', features, reg3(0.2))
# regressor_model('Formation energy [eV/atom]', 'Formation Energy', features, vote2(c=23.36, g=0.08, w=0.7, lr=0.25))

print("Modeling for Vacancy Energy")
# regressor_model('Vacancy energy [eV/O atom]', 'Vacancy Energy', features, reg1(c=31.41, g=0.179))
# regressor_model('Vacancy energy [eV/O atom]', 'Vacancy Energy', features, reg2)
# regressor_model('Vacancy energy [eV/O atom]', 'Vacancy Energy', features, reg3(0.25))
# regressor_model('Vacancy energy [eV/O atom]', 'Vacancy Energy', features, vote3(c=31.41, g=0.179, w=0.5, lr=0.25))
regressor_model('Vacancy energy [eV/O atom]', 'Vacancy Energy', features, vote2(c=31.41, g=0.179, w=0.7, lr=0.25))

