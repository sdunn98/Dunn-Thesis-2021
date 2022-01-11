import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics

# ---------------------- READ DATASETS ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
importances = 'datasets_processed/feature-importance.csv'

df = pd.read_csv(dataset)
importance_df = pd.read_csv(importances)

features = list(importance_df['Band gap [eV]'])


todrop = []
for i in df.columns:
    if i not in features[:25]:
        todrop.append(i)

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
clf2 = KNeighborsClassifier(n_neighbors=3, p=3)
clf3 = DecisionTreeClassifier()
clf4 = GradientBoostingClassifier()

# model = VotingClassifier(estimators=[('rfc', clf1), ('knc', clf2), ('dtc', clf3), ('gbc', clf4)])
model = clf1

model.fit(X_train, Y_train)
prediction = model.predict(X_test)
report = metrics.classification_report(Y_test, prediction)
accuracy = metrics.accuracy_score(Y_test, prediction)
R2 = metrics.r2_score(Y_test, prediction)
BAS = metrics.balanced_accuracy_score(Y_test, prediction)
precision = metrics.precision_score(Y_test, prediction, average=None)


print(f"Precision score (zero prediction) = {precision[0]:.4f}")
print(f"Accuracy score = {accuracy:.4f}")
# print(f"R2 = {R2:.3f}")
print(f"Balanced accuracy score = {BAS:.4f}")
print(report)

metrics.plot_confusion_matrix(model, X_test, Y_test, normalize='pred', values_format='.3f', cmap="GnBu")
plt.xlabel("Predicted Band Gap")
plt.ylabel("Actual Band Gap")
plt.title("K-Neighbours Classifier Confusion Matrix")
plt.tight_layout()
plt.xticks([0,1],["Zero", "Non-zero"], fontsize='small')
plt.yticks([0,1],["Zero", "Non-zero"], rotation=90, verticalalignment="center", fontsize='small')
# plt.savefig('figures/BG-confusion-matrix', dpi=300)

precision_loss = []

for i in range(len(list(Y_test))):
    if list(Y_test)[i] != list(prediction)[i] and list(prediction)[i] == 0:
        precision_loss.append(list(df['Band gap [eV]'])[-len(Y_test):][i])

print(f"Max prediction error = {max(precision_loss)}")
print(f"Average prediction error = {sum(precision_loss)/len(precision_loss)}\n")




