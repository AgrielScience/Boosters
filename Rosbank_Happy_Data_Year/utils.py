import pandas as pd
import reverse_geocoder as rg
from math import sin, cos, sqrt, atan2, radians
from yandex_geocoder import Client
from sklearn.metrics import mean_squared_error
from sklearn import cluster, mixture
import numpy as np
import colorama
from hyperopt import STATUS_OK
from catboost import *
from sklearn.model_selection import KFold

def parse_venue(venue):
    primary_category = [d for d in venue['categories'] if d['primary'] == True]

    if len(primary_category) > 0:
        primary_category = primary_category.pop()
    else:
        primary_category = None

    venue_d = {
        'id': venue['id'],
        'name': venue['name'],
        'address': venue['location']['address'] if 'address' in venue['location'] else None,
        'lat': venue['location']['lat'],
        'lng': venue['location']['lng'],
        'primary_category': primary_category['name'] if primary_category else None,
        'primary_category_id': primary_category['id'] if primary_category else None,
        'checkins': venue['stats']['checkinsCount'],
        'distance': venue['location']['distance']
    }

    return venue_d

def search_venues(fq, lat, long, limit=1, radius = 100000, intent='browse', categoryId = '4bf58dd8d48988d1eb931735'):
    return fq.venues.search(params={'ll': '%s,%s' % (lat, long), 'limit': limit, 'radius': radius, 'intent': intent,
                                   'categoryId': categoryId})

def get_coords(x, y):
    if pd.isnull(x) | pd.isnull(y):
        return 'nan'
    else:
        coordinates = (x, y)
        results = rg.search(coordinates)
        return results

def distance(x1,y1,x2,y2):
    R = 6373.0
    dlon = y2 - y1
    dlat = x2 - x1
    a = sin(dlat/2)**2 + cos(x1) * cos(x2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def distance_rad(x, y):
    """
    Параметры
    ----------
    x : tuple, широта и долгота первой геокоординаты
    y : tuple, широта и долгота второй геокоординаты

    Результат
    ----------
    result : дистанция в километрах между двумя геокоординатами
    """
    R = 6373.0
    lat_a, long_a, lat_b, long_b = map(radians, [*x, *y])
    dlon = long_b - long_a
    dlat = lat_b - lat_a
    a = sin(dlat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_coords_yandex(address):
    try:
        return Client.coordinates(address)
    except:
        return 'coords_nan'

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def get_cities_info(data, cities_list):
    data['city'] = data.address_rus.apply(lambda x: x.split(',')[2])
    rare_cities_geo = data.geo_city.value_counts()[(data.geo_city.value_counts() < 20) == True].index
    data['city_type'] = data.geo_city.apply(lambda x: 'RARE' if x in rare_cities_geo else x)
    data['street'] = data[~data.address_rus.isnull()].address_rus.apply(lambda x: x.split(',')[0])
    data['street'] = data['geo_city'].astype(str) + '_' + data['street'].astype('str')
    data['geo_city_rank'] = data.geo_city.rank().fillna(-1)
    data['geo_city_type_rank'] = data.geo_city_rank.rank().fillna(-1)
    data['street'] = data['street'].str.lower().replace(' ', '')
    rare_cities = data.city.value_counts()[(data.city.value_counts() < 20) == True].index
    data.city = data.city.apply(lambda x: 'RARE' if x in rare_cities else x)
    data['city_rank'] = data.city.rank().fillna(-1)
    data['city_type_rank'] = data.city_type.rank().fillna(-1)
    cities_list = cities_list[['Регион', 'Город', 'Федеральный округ', 'Население']].rename(
        columns={'Город': 'city', 'Федеральный округ': 'okrug', 'Население': 'popularity'})
    cities_list['city'] = np.where(cities_list.city.isnull(), cities_list['Регион'], cities_list.city)
    cities_list = cities_list[['city', 'okrug', 'popularity']]
    cities_list.city = cities_list.city.str.lower().replace(' ', '')
    cities_list = cities_list.drop_duplicates(subset='city', keep="first")
    data.city = data.city.str.lower().replace(' ', '')
    data['city'] = data.city.apply(lambda x: x.strip())
    data = pd.merge(data, cities_list, how='left', on='city')
    data.popularity = np.where(data.popularity.isnull(), 0, data.popularity)
    data.popularity = data.popularity.astype(int)
    data['popularity_type'] = np.where((data.popularity > 50000) & (data.popularity < 100000), 'middle',
                                       np.where((data.popularity > 100000) & (data.popularity < 250000), 'big',
                                                np.where((data.popularity > 250000) & (data.popularity < 500000),
                                                         'huge',
                                                         np.where(
                                                             (data.popularity > 500000) & (data.popularity < 1000000),
                                                             'superbig',
                                                             np.where(data.popularity > 1000000, 'colossal', 'rare')))))
    data.okrug = np.where(data.okrug.isnull(), 'rare', data.okrug)
    data = data.drop('popularity', 1)
    return data


def cluster_model(newdata, data, model_name, input_param):
    ds = data
    params = input_param
    if str.lower(model_name) == 'kmeans':
        cluster_obj = cluster.KMeans(n_clusters=params['n_clusters'])
    if str.lower(model_name) == str.lower('MiniBatchKMeans'):
        cluster_obj = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    if str.lower(model_name) == str.lower('SpectralClustering'):
        cluster_obj = cluster.SpectralClustering(n_clusters=params['n_clusters'])
    if str.lower(model_name) == str.lower('MeanShift'):
        cluster_obj = cluster.MeanShift(bandwidth=params['bandwidth'])
    if str.lower(model_name) == str.lower('DBSCAN'):
        cluster_obj = cluster.DBSCAN(eps=params['eps'])
    if str.lower(model_name) == str.lower('AffinityPropagation'):
        cluster_obj = cluster.AffinityPropagation(damping=params['damping'],
                                                  preference=params['preference'])
        cluster_obj.fit(ds)
    if str.lower(model_name) == str.lower('Birch'):
        cluster_obj = cluster.Birch(n_clusters=input_param['n_clusters'])
    if str.lower(model_name) == str.lower('GaussianMixture'):
        cluster_obj = mixture.GaussianMixture(n_components=params['n_clusters'],
                                              covariance_type='full')
        cluster_obj.fit(ds)

    if str.lower(model_name) in ['affinitypropagation', 'gaussianmixture']:
        model_result = cluster_obj.predict(ds)
    else:
        model_result = cluster_obj.fit_predict(ds)

    newdata[model_name] = pd.DataFrame(model_result)

    return newdata

def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.quantile(0.75)

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['rsm'] = space['rsm']
    return params


def objective(space, folds, test, X, Y, X_test, cat_features):
    n_fold = folds
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
    global obj_call_count, cur_best_loss
    obj_call_count += 1
    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count, cur_best_loss))
    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count, cur_best_loss))
    params = get_catboost_params(space)
    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str))

    model = CatBoostRegressor(iterations=5000,
                              learning_rate=params['learning_rate'],
                              depth=int(params['depth']),
                              loss_function='RMSE',
                              use_best_model=True,
                              l2_leaf_reg=params['l2_leaf_reg'],
                              early_stopping_rounds=25,
                              random_seed=42,
                              verbose=False
                              )
    # eval_set  = [(X_train, Y_train), (X_valid, Y_valid)]

    prediction = np.zeros(test.shape[0])
    rmse_list = []
    for fold_n, (train_index, test_index) in enumerate(folds.split(X)):
        print('Fold:', fold_n)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(X_train, Y_train, cat_features=cat_features,
                  eval_set=[(X_valid, Y_valid)], verbose=False)

        y_pred = model.predict(X_test, ntree_end=model.best_iteration_)
        prediction += y_pred
        rmse_fold = rmse(Y_valid, model.predict(X_valid, ntree_end=model.best_iteration_))
        rmse_list.append(rmse_fold)

    rmse_loss = sum(rmse_list) / len(rmse_list)
    print('Iteration RMSE:{}'.format(rmse_loss))

    if rmse_loss < cur_best_loss:
        cur_best_loss = rmse_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

    return {'loss': rmse_loss, 'status': STATUS_OK}