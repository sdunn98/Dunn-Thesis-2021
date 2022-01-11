import pandas as pd
import numpy as np

# ---------------------- READ DATASETS ---------------------- #
mendeleev_loc = 'datasets_original/atom_char_mendeleev.csv'
perovskite_loc = 'datasets_original/database_perovskite.csv'

atom_char_df = pd.read_csv(mendeleev_loc).set_index('symbol')
perovskite_df = pd.read_csv(perovskite_loc)

# Save parameter controls whether datasets are exported to CSV
save = True

# ------------------------- FUNCTIONS ------------------------ #

# Clean data - remove incomplete data
def clean(dataset):
    dataset = dataset.replace('-', np.NaN)
    dataset = dataset.dropna()
    dataset = dataset.apply(pd.to_numeric, errors='ignore')
    return dataset

# Save to CSV file
def savecsv(dataset, name, directory='datasets_processed/', index=False):
    save_string = directory + name + '.csv'
    dataset.to_csv(save_string, index=index)
    print('CSV file has been saved to... {}'.format(save_string))

# ----------------------- PRE-PROCESSING ---------------------- #

perovskite_df = clean(perovskite_df)

# Atom characteristics to include
mendeleev_chars = ['atomic_number', 'atomic_radius', 'atomic_volume', 'boiling_point', 'density',
                   'dipole_polarizability', 'lattice_constant', 'melting_point', 'specific_heat',
                   'vdw_radius', 'covalent_radius_cordero', 'covalent_radius_pyykko', 'en_pauling',
                   'heat_of_formation', 'covalent_radius_slater', 'vdw_radius_uff', 'vdw_radius_mm3',
                   'abundance_crust', 'en_ghosh', 'vdw_radius_alvarez', 'c6_gb', 'atomic_weight',
                   'atomic_radius_rahm', 'covalent_radius_pyykko_double', 'mendeleev_number',
                   'pettifor_number', 'glawe_number']

# Remove other columns
for char in atom_char_df.columns:
    if char not in mendeleev_chars:
        atom_char_df = atom_char_df.drop([char], axis='columns')


# Create new dataset
df = pd.DataFrame()

# A & B elements
df['A'] = perovskite_df['A']
df['B'] = perovskite_df['B']

# Target perovskite properties
df['Formation energy [eV/atom]'] = perovskite_df['Formation energy [eV/atom]']
df['Vacancy energy [eV/O atom]'] = perovskite_df['Vacancy energy [eV/O atom]']
df['Band gap [eV]'] = perovskite_df['Band gap [eV]']
df.reset_index(inplace=True, drop=True)

# Include atom characteristics of A & B site atoms
for char in atom_char_df.columns:
    for site in ['A','B']:
        values = [atom_char_df.loc[element][char] for element in df[site]]
        char_title = char.replace('_', ' ')
        df['{} {}'.format(site, char_title)] = values

# Get scaler values
char_names = []
scalers = []

for i in atom_char_df.columns:
    char_names.append(i.replace('_',' '))
    minn = atom_char_df[i].min()
    maxx = atom_char_df[i].max()
    scalers.append([minn,maxx])

targets = ['Formation energy [eV/atom]', 'Vacancy energy [eV/O atom]', 'Band gap [eV]']

for i in targets:
    char_names.append(i)
    minn = df[i].min()
    maxx = df[i].max()
    scalers.append([minn, maxx])

# Create dataframe of scaler values and save to CSV
scaler_df = pd.DataFrame(scalers, index=char_names, columns=['Min', 'Max'])
scaler_df.index.rename('Feature')

df = clean(df)

# Save to CSV file
if save == True:
    savecsv(df, 'db-primary')

# Group atom characteristics for normalisation
char_groups = [ ['Formation energy [eV/atom]'],
                ['Vacancy energy [eV/O atom]'],
                ['Band gap [eV]'],
                ['A atomic number', 'B atomic number'],
                ['A atomic radius', 'B atomic radius'],
                ['A atomic volume', 'B atomic volume'],
                ['A boiling point', 'B boiling point'],
                ['A density', 'B density'],
                ['A dipole polarizability', 'B dipole polarizability'],
                ['A lattice constant', 'B lattice constant'],
                ['A melting point', 'B melting point'],
                ['A specific heat', 'B specific heat'],
                ['A vdw radius', 'B vdw radius'],
                ['A covalent radius cordero', 'B covalent radius cordero'],
                ['A covalent radius pyykko', 'B covalent radius pyykko'],
                ['A en pauling', 'B en pauling'],
                ['A heat of formation', 'B heat of formation'],
                ['A covalent radius slater', 'B covalent radius slater'],
                ['A vdw radius uff', 'B vdw radius uff'],
                ['A vdw radius mm3', 'B vdw radius mm3'],
                ['A abundance crust', 'B abundance crust'],
                ['A en ghosh', 'B en ghosh'],
                ['A vdw radius alvarez', 'B vdw radius alvarez'],
                ['A c6 gb', 'B c6 gb'],
                ['A atomic weight', 'B atomic weight'],
                ['A atomic radius rahm', 'B atomic radius rahm'],
                ['A covalent radius pyykko double', 'B covalent radius pyykko double'],
                ['A mendeleev number', 'B mendeleev number'],
                ['A pettifor number', 'B pettifor number'],
                ['A glawe number', 'B glawe number'] ]

# Normalise features by group
df = df.drop(['A','B'], axis='columns')

group_names = []

for group in char_groups:
    name = df[group].columns[0]
    if name[0] == 'A':
        name = name[2:]
    group_names.append(name)

    norm_values = []
    minn = scaler_df.loc[name]['Min']
    maxx = scaler_df.loc[name]['Max']

    for i in df[group].values:
        i = (i - minn)/(maxx - minn)
        norm_values.append(i)

    df[group] = norm_values

# Save normalised dataset to csv
if save == True:
    savecsv(df, 'db-norm')

# Save scaler values to CSV
if save == True:
    savecsv(scaler_df, 'norm-scalers', index=True)

# Save a shuffled version of normalised dataframe
shuffled_df = df.sample(frac=1)
if save == True:
    savecsv(shuffled_df, 'db-norm-shuffled')

# Save modified atom characteristics dataset
if save == True:
    savecsv(atom_char_df, 'db-atoms', index=True)