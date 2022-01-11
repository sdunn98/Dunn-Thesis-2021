import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------- READ DATASET ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
df = pd.read_csv(dataset)


# ------------------- FEATURE IMPORTANCE ------------------- #
def importance(target: str, label: str, saveimg=True):
    # 'target' must be one of 'Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', or 'Band gap [eV]'

    X = df.drop(['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'], axis='columns')
    Y = df[target]

    if target == 'Band gap [eV]':
        classes = []
        for i in Y:
            if i == 0:
                classes.append(0)
            else:
                classes.append(1)

        Y = classes

    # Split into train/test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

    # Random forest regressor
    if target == 'Band gap [eV]':
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    model.fit(X_train, Y_train)

    # Feature Importances
    importances = model.feature_importances_

    feature_names = []
    for i in X.columns:
        feature_names.append(i)

    forest_importances = pd.Series(importances, index=feature_names)
    sorted_importances = forest_importances.sort_values(ascending=True)

    plt.subplots(figsize=(7, 10), dpi=300)
    sorted_importances.plot.barh()
    plt.title(f"Feature Importances ({label})", fontsize='x-large')
    plt.xlabel("Mean Decrease in Impurity (MDI)", fontsize='medium')
    plt.tight_layout()
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')

    savename = "feature-importance-"
    words = label.split()
    for word in words:
        savename += word[0].upper()

    if saveimg == True:
        plt.savefig('figures/feature_importance/' + savename + '.png')
        print(f"Feature Importance ({label}) chart saved to... figures/feature_importance/{savename}.png")

    # plt.show()

    names = list(forest_importances.index)
    values = list(forest_importances.values)
    asc_names = sorted_importances.index.tolist()

    names.reverse()
    values.reverse()

    return names, values, asc_names


FE_names, FE_values, FE_asc = importance('Formation energy [eV/atom]', 'Formation Energy')
VE_names, VE_values, VE_asc = importance('Vacancy energy [eV/O atom]', 'Vacancy Energy')
BG_names, BG_values, BG_asc = importance('Band gap [eV]', 'Band Gap')

# Plot feature importances side-by-side
plt.figure(figsize=(10, 10), dpi=300)
plt.suptitle('Feature Importance by Target Property', fontsize='x-large')

plt.subplot(1,3,1)
plt.barh(FE_names, FE_values)
plt.title('Formation Energy', fontsize='medium')
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='small')
plt.xlabel('Mean Decrease in Impurity (MDI)', fontsize='small')
plt.xlim(right=0.25)
plt.tight_layout()

plt.subplot(1,3,2)
plt.barh(VE_names, VE_values)
plt.title('Vacancy Energy', fontsize='medium')
plt.xticks(fontsize='x-small')
plt.yticks([])
plt.xlabel('Mean Decrease in Impurity (MDI)', fontsize='small')
plt.xlim(right=0.25)
plt.tight_layout()

plt.subplot(1,3,3)
plt.barh(BG_names, BG_values)
plt.title('Band Gap', fontsize='medium')
plt.xticks(fontsize='x-small')
plt.yticks([])
plt.xlabel('Mean Decrease in Impurity (MDI)', fontsize='small') 
plt.xlim(right=0.25)
plt.tight_layout()

plt.savefig('figures/feature_importance/feature-importance.png')
print(f"Feature Importance chart saved to... figures/feature_importance/feature-importance.png")

# Create dataframe to store features in order of increasing importance for each target
importance_df = pd.DataFrame()
importance_df['Formation energy [eV/atom]'] = FE_asc
importance_df['Vacancy energy [eV/O atom]'] = VE_asc
importance_df['Band gap [eV]'] = BG_asc

importance_df.to_csv('datasets_processed/feature-importance.csv', index=True)
print('CSV file has been saved to... datasets_processed/feature-importance.csv')