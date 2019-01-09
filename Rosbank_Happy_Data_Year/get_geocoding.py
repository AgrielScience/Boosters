from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from utils import *

data = pd.read_csv("data/data_with_places.csv")
time_df = pd.read_csv('data/work_time_atm.csv')

new_df = data.copy()
new_df['coords'] = new_df.apply(lambda x: get_coords(x.lat, x.long), axis=1)
new_df = new_df.reset_index().drop('index', 1)
new_df['coords'] = new_df['coords'].astype(str)
new_df['city_lat'] = new_df.coords.apply(lambda x: x.split(',')[1])
new_df['city_long'] = new_df.coords.apply(lambda x: x.split(',')[3])
new_df['geo_city'] = new_df.coords.apply(lambda x: x.split(',')[5])
new_df['region'] = new_df.coords.apply(lambda x: x.split(',')[7])
new_df['city_lat'] = new_df['city_lat'].astype(str).apply(lambda x: x.replace("'", ''))
new_df['city_lat'] = new_df['city_lat'].astype(str).apply(lambda x: x.replace(")", ''))
new_df['city_long'] = new_df['city_long'].astype(str).apply(lambda x: x.replace("'", ''))
new_df['city_long'] = new_df['city_long'].astype(str).apply(lambda x: x.replace(")", ''))
new_df['geo_city'] = new_df['geo_city'].astype(str).apply(lambda x: x.replace("'", ''))
new_df['geo_city'] = new_df['geo_city'].astype(str).apply(lambda x: x.replace(")", ''))
new_df['region'] = new_df['region'].astype(str).apply(lambda x: x.replace("'", ''))
new_df['region'] = new_df['region'].astype(str).apply(lambda x: x.replace(")", ''))
new_df['city_lat'] = new_df['city_lat'].astype(float)
new_df['city_long'] = new_df['city_long'].astype(float)
new_df['diff_from_centre'] = new_df.apply(lambda x: distance(x.lat, x.long, x.city_lat, x.city_long), axis=1)
new_df = new_df.drop('coords', 1)
distances = pd.DataFrame(cdist(new_df[['lat', 'long']].values, new_df[['lat', 'long']].values, distance_rad),
                         index=new_df.id, columns=new_df.id)
dist_all = distances.copy().reset_index()
res_id = dist_all[['id']]
dist_25m = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 0.025, x, None))
dist_25m = dist_25m.count(axis=1)
res_id['count_25m'] = dist_25m
dist_50m = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 0.05, x, None))
dist_50m = dist_50m.count(axis=1)
res_id['count_50m'] = dist_50m
dist_100m = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 0.1, x, None))
dist_100m = dist_100m.count(axis=1)
res_id['count_100m'] = dist_100m
dist_250m = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 0.25, x, None))
dist_250m = dist_250m.count(axis=1)
res_id['count_250m'] = dist_250m
dist_500m = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 0.5, x, None))
dist_500m = dist_500m.count(axis=1)
res_id['count_500m'] = dist_500m
dist_750m = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 0.75, x, None))
dist_750m = dist_750m.count(axis=1)
res_id['count_750m'] = dist_750m
dist_1 = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 1, x, None))
dist_1 = dist_1.count(axis=1)
res_id['count_1'] = dist_1
dist_2 = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 2, x, None))
dist_2 = dist_2.count(axis=1)
res_id['count_2'] = dist_2
dist_3 = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 3, x, None))
dist_3 = dist_3.count(axis=1)
res_id['count_3'] = dist_3
dist_4 = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 4, x, None))
dist_4 = dist_4.count(axis=1)
res_id['count_4'] = dist_4
dist_5 = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 5, x, None))
dist_5 = dist_5.count(axis=1)
res_id['count_5'] = dist_5
dist_10 = dist_all.drop('id', 1).applymap(lambda x: np.where(x <= 10, x, None))
dist_10 = dist_10.count(axis=1)
res_id['count_10'] = dist_10
new_df = pd.merge(new_df, res_id, how='left', on='id')

time_df = time_df.rename(columns={'Номер банкомата':'atm_bank', 'Регион':'region',
                                 'Город':'city', 'Адрес':'adress', 'Режим работы':'work_time'})
time_df['has_city'] = time_df.adress.apply(lambda x: np.where(('Город') in x, 1, 0))
time_df_city = time_df[time_df.has_city == 1]
time_df_city.adress = time_df_city.adress.apply(lambda x: x.split('Город')[1])
time_df_city.adress = time_df_city.adress.str.replace('Дом', '')
time_df_city['coords'] = time_df_city.adress.apply(lambda x: get_coords_yandex(x))
res_df = time_df_city.copy()
res_df = res_df[res_df.coords != 'coords_nan']
res_df['coords'] = res_df.coords.astype(str)
res_df['city_lat'] = res_df.coords.apply(lambda x: x.split(',')[1])
res_df['city_long'] = res_df.coords.apply(lambda x: x.split(',')[0])
res_df['city_lat'] = res_df.city_lat.str.replace("'","")
res_df['city_lat'] = res_df.city_lat.str.replace(')', '')
res_df['city_long'] = res_df.city_long.str.replace("'","")
res_df['city_long'] = res_df.city_long.str.replace('(', '')
res_df['city_lat'] = res_df.city_lat.astype(float)
res_df['city_long'] = res_df.city_long.astype(float)
knc = KNeighborsClassifier(metric=distance)
dots = res_df[['city_lat','city_long']].dropna()
knc.fit(X=dots , y=np.ones(dots.shape[0]))
distances, indexes = knc.kneighbors(X=new_df[['lat', 'long']],n_neighbors=1)
new_df['distance_nearest'] = distances
new_df['indexes_nearest'] = indexes
res_df['indexes_nearest'] = res_df.index
new_df = pd.merge(new_df, res_df[['atm_bank', 'work_time', 'indexes_nearest']], how='left', on='indexes_nearest')
new_df.to_csv('data/data_with_geo_no_nans.csv', index=False)