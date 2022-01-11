# Test random forest classifier for Band Gap - not as effective as RFR

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics

# ---------------------- READ DATASETS ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
importances = 'datasets_processed/feature-importance.csv'

df = pd.read_csv(dataset)
importance_df = pd.read_csv(importances)

sorted_importance = list(importance_df['Band gap [eV]'])

def reduction(classifier):

    num = 0
    scores = []
    features = []

    while num < len(sorted_importance):

        todrop = ['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]'] + sorted_importance[:num]

        X = df.drop(todrop, axis='columns')
        print("Number of features = " + str(len(X.columns)))
        classes = []
        for i in df['Band gap [eV]']:
            if i == 0:
                classes.append(0)
            else:
                classes.append(1)

        Y = classes


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        clf1 = RandomForestClassifier()
        clf2 = KNeighborsClassifier()
        clf4 = GradientBoostingClassifier()

        if classifier == 'RFC':
            model = clf1
        elif classifier == 'KNC':
            model = clf2
        elif classifier == 'GBC':
            model = clf4

        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)

        accuracy = metrics.accuracy_score(Y_test, prediction)

        print(f"Accuracy score = {accuracy:.4f}")

        scores.append(accuracy)
        features.append(len(sorted_importance) - num)
        num += 1

    return features, scores


features_1, scores_1 = reduction('RFC')
features_2, scores_2 = reduction('GBC')
features_3, scores_3 = reduction('KNC')

plt.figure(figsize=(7,5), dpi=300)
plt.plot(features_1, scores_1, marker='o', markersize=3)
plt.plot(features_2, scores_2, marker='o', markersize=3)
plt.plot(features_3, scores_3, marker='o', markersize=3)
plt.title(f'Classifier Model Feature Reduction - Band Gap', fontsize='x-large')

plt.xlabel('Number of Features', fontsize='large')
plt.ylabel('Accuracy', fontsize='large')
plt.grid(True)
plt.xticks(fontsize='medium')
plt.xlim(left=0)
plt.yticks(fontsize='medium')
plt.ylim(top=1, bottom=0)
plt.legend(['RFC', 'GBC', 'KNC'], loc='lower right')
plt.tight_layout()

plt.savefig(f'figures/feature_reduction/BG-feature-reduction.png')













