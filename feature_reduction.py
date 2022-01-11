import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn import metrics
from sklearn.model_selection import train_test_split

# ---------------------- READ DATASETS ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
importances = 'datasets_processed/feature-importance.csv'

df = pd.read_csv(dataset)
importance_df = pd.read_csv(importances)


saveimg = True
plotimg = True

# -------------------- FEATURE REDUCTION -------------------- #
def reduction(target: str, label: str, regressor: str):
    print(f"\nFeature Reduction for {label} using {regressor}:")

    # Feature Importance
    sorted_importance = list(importance_df[target])

    num = 0
    scores = []
    features = []
    Y = df[target]

    while num < len(sorted_importance):

        # Feature Reduction
        todrop = ['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'] + sorted_importance[:num]
        # todrop = ['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]']
        X = df.drop(todrop, axis='columns')

        # RFR model
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

        if regressor == 'RFR':
            model = RandomForestRegressor()
        elif regressor == 'SVR':
            model = SVR()
        elif regressor == 'GBR':
            model = GradientBoostingRegressor()
        elif regressor == 'DTR':
            model = DecisionTreeRegressor()
        elif regressor == 'KNR':
            model = KNeighborsRegressor()
        elif regressor == 'RNR':
            model = RadiusNeighborsRegressor()
        elif regressor == 'NuSVR':
            model = NuSVR()
        elif regressor == 'MLP':
            model = MLPRegressor()
        else:
            print("Please specify a regressor model - one of 'RFR', 'SVR', 'KernelRidge'")
            exit()

        model.fit(X_train, Y_train)

        # Prediction
        prediction = model.predict(X_test)
        R2 = metrics.r2_score(Y_test, prediction)
        RMSE = metrics.mean_squared_error(Y_test, prediction)
        MAE = metrics.mean_absolute_error(Y_test, prediction)

        print(f'Features dropped = {num}')
        print(f'R^2 = {R2:0.4f}')
        # print(f'RMSE = {RMSE:0.4f}')
        # print(f'MAE = {MAE:0.4f}')
        # print(f'{R2:0.4f}')
        # print(f'{RMSE:0.4f}')
        # print(f'{MAE:0.4f}')

        scores.append(R2)
        features.append(len(sorted_importance)-num)
        num += 1

    return features, scores

# features_1, scores_1 = reduction('Formation energy [eV/atom]', 'Formation Energy', 'RFR')
# features_2, scores_2 = reduction('Formation energy [eV/atom]', 'Formation Energy', 'NuSVR')
# features_3, scores_3 = reduction('Formation energy [eV/atom]', 'Formation Energy', 'GBR')

features_1, scores_1 = reduction('Vacancy energy [eV/O atom]', 'Vacancy Energy', 'RFR')
features_2, scores_2 = reduction('Vacancy energy [eV/O atom]', 'Vacancy Energy', 'NuSVR')
features_3, scores_3 = reduction('Vacancy energy [eV/O atom]', 'Vacancy Energy', 'GBR')


if plotimg == True:
    # Plot number of input features vs R^2 performance
    plt.figure(figsize=(7,5), dpi=300)
    plt.plot(features_1, scores_1, marker='o', markersize=3)
    plt.plot(features_2, scores_2, marker='o', markersize=3)
    plt.plot(features_3, scores_3, marker='o', markersize=3)
    # plt.title('RFR Feature Reduction', fontsize='x-large')
    plt.title(f'Regressor Model Feature Reduction - Vacancy Energy', fontsize='x-large')

    plt.xlabel('Number of Features', fontsize='large')
    plt.ylabel('R^2', fontsize='large')
    plt.grid(True)
    plt.xticks(fontsize='medium')
    plt.xlim(left=0)
    plt.yticks(fontsize='medium')
    plt.ylim(top=1, bottom=0)
    plt.legend(['RFR', 'NuSVR', 'GBR'], loc='lower right')
    plt.tight_layout()

if saveimg == True:
    # plt.savefig(f'figures/feature_reduction/FE-feature-reduction.png')
    # print(f'Feature Reduction chart saved to... figures/feature_reduction/FE-feature-reduction.png')

    plt.savefig(f'figures/feature_reduction/VE-feature-reduction.png')
    print(f'Feature Reduction chart saved to... figures/feature_reduction/VE-feature-reduction.png')
else:
    plt.show()