import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------- READ DATASET ---------------------- #
dataset = 'datasets_processed/db-norm-shuffled.csv'
df = pd.read_csv(dataset)

# Save parameter controls whether images are saved to file
save = False

# ------------------ FEATURE CORRELATION ------------------ #
corrMatrix = df.corr()
plt.figure(figsize=(14,12), dpi=300)
sn.heatmap(corrMatrix, annot=False, cmap="viridis")

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Feature Correlation", fontsize=24, pad=12)
plt.tight_layout()

# -------------------- SAVE/SHOW PLOT -------------------- #
if save == True:
    plt.savefig('figures/feature-correlation.png')
    print('Feature correlation chart has been saved to... figures/feature-correlation.png')

# plt.show()


importance_df = pd.read_csv('datasets_processed/feature-importance.csv')

# features_FE = list(importance_df['Formation energy [eV/atom]'])
features_FE = list(importance_df['Vacancy energy [eV/O atom]'])

features_FE.reverse()
print(features_FE)
features_corr = [0]

for i in range(1, len(features_FE)):
    max_corr = 0
    for j in range(0, i):
        corr = abs(corrMatrix[features_FE[i]][features_FE[j]])
        if corr > max_corr:
            max_corr = corr
    features_corr.append(max_corr)


features = []

for i in range(len(features_FE)):
    if features_corr[i] < 0.9:
        features.append(features_FE[i])

print(features)
print(len(features))