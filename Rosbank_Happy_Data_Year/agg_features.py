from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from utils import *
import pandas as pd

train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
train['isTrain'] = True
test['isTrain'] = False
data = pd.read_csv('data/data_with_geo_no_nans.csv')
cities_list = pd.read_csv('data/cities.csv')

#Calculate distances from sample to sample and get nearest neighbors

knc = KNeighborsClassifier(metric=distance_rad)
dots = data[['lat','long']].dropna()
knc.fit(X=dots , y=np.ones(dots.shape[0]))
distances, indexes = knc.kneighbors(X=dots,n_neighbors=10)

for i in range(1, 6):
    dots['distance_%s' % i] = distances[:, i]

#Calculate statistics from distances

dots['mean'] = dots.iloc[:, dots.columns.str.contains('distance')].mean(axis=1)
dots['median'] = dots.iloc[:, dots.columns.str.contains('distance')].median(axis=1)
dots['std'] = dots.iloc[:, dots.columns.str.contains('distance')].std(axis=1)
dots['q1'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.1, axis=1)
dots['q2'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.2, axis=1)
dots['q3'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.3, axis=1)
dots['q4'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.4, axis=1)
dots['q5'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.5, axis=1)
dots['q6'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.6, axis=1)
dots['q7'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.7, axis=1)
dots['q8'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.8, axis=1)
dots['q9'] = dots.iloc[:, dots.columns.str.contains('distance')].quantile(.9, axis=1)

dots_pca = data[['lat', 'long']].dropna()
pca = PCA(n_components=1)
dots['coords_pca'] = pca.fit_transform(dots_pca)
data = pd.concat([data, dots.drop(['lat', 'long'], 1)], axis=1)

#Get city info and clusters from data

data = get_cities_info(data, cities_list)
cluster_list = ["KMeans", "MiniBatchKMeans", "DBSCAN", "Birch", "MeanShift"]
input_param = {'n_clusters':140, 'bandwidth':0.1, "damping":0.9, "eps":1, 'min_samples':3,
               "preference":-200}
pca_comp = data[['lat','long']].fillna(0)

for i in cluster_list:
    data = cluster_model(data, pca_comp, i, input_param)

data['avg_test'] = np.where(data.target.isnull(), data.target.mean()*(-1), data.target.mean())
target = data[['target']]
data = data.drop('target', 1)
data['avg_target_atm'] = data.groupby('atm_group')['avg_t'].transform(np.mean)
data['med_target_atm'] = data.groupby('atm_group')['avg_t'].transform(np.median)
data['std_target_atm'] = data.groupby('atm_group')['avg_t'].transform(np.std)
data = data.fillna(0)

# Calculate distances from sample to group

group_df = data[['lat', 'long', 'atm_group']].copy()
concatcoords = group_df.groupby(['atm_group'])
union = pd.DataFrame()
kneighborsdf = pd.DataFrame()
group_list = data.atm_group.drop_duplicates().reset_index().drop('index', 1)
new_data = data.copy()
indesx = 0
for name, group in concatcoords:
    localgroup = group_df[group_df['atm_group'] == name]
    knc = KNeighborsClassifier(metric=distance)
    knc.fit(X=localgroup[['lat', 'long']], y=np.ones(localgroup.shape[0]))
    union_knn = pd.DataFrame()
    for sample in group_list.atm_group:
        groupcoords = group_df[group_df['atm_group'] == sample]
        distances, ids = knc.kneighbors(X=groupcoords[['lat', 'long']], n_neighbors=10)
        kneighborsdf = pd.DataFrame()

        for i in range(1, 10):
            kneighborsdf[('distancegroup_{}_{}').format(indesx, i)] = distances[:, i]
            # kneighborsdf[('indexesegroup_{}_{}').format(indesx, i)] = ids[:,i]

        union_knn = union_knn.append(kneighborsdf)

    union = pd.concat([union, union_knn], 1)
    indesx += 1

union = union.reset_index()
new_data = pd.concat([new_data, union], 1)

#Calculate statistics from group distances

categories = ['metro_distance', 'air_distance', 'train_distance', 'bus_distance', 'tram_distance']

for c in categories:
    new_data[c+'mean'] = new_data.groupby('atm_group')[c].transform(np.mean)
    new_data[c+"median"] = new_data.groupby('atm_group')[c].transform(np.median)
    new_data[c+"std"] = new_data.groupby('atm_group')[c].transform(np.std)
    new_data['atm_grp_'+str(c)] = data.groupby('atm_group')[c].transform(q1)
    new_data['atm_grp_'+str(c)] = data.groupby('atm_group')[c].transform(q2)

new_data['target'] = target.target
new_data.to_csv("data/prepared_data.csv", index=False)