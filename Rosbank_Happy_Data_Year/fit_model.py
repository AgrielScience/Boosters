from hyperopt import hp, tpe, Trials
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Should use HYPEROPT? (T/F)')

parser.add_argument('-h', '--hyperopt', required=False)

args = parser.parse_args()

hyperopt = args.hyperopt

data = pd.read_csv('data/prepared_data.csv')
test = pd.read_csv('data/test.csv')

features = ["id", "atm_group", "city", "city_type","street", "geo_city", "KMeans", "MiniBatchKMeans", "DBSCAN",
            "Birch", "MeanShift", "primary_category", "is_metro", "popularity_type", "okrug","atm_bank","work_time"]

for f in features:
    data[f] = data[f].rank().fillna(-1)
    data[f] = data[f].values.astype('str')

X = data[data.isTrain == True].drop(['target', 'isTrain', 'address', 'address_rus'], 1)
Y = data[data.isTrain == True][['target']]
X_test = data[data.isTrain == False].drop(['target', 'isTrain', 'address', 'address_rus'], 1)

cat_features = []
for i in features:
    cf = X.columns.get_loc(i)
    cat_features.append(cf)

if hyperopt=="T":
    # кол-во случайных наборов гиперпараметров
    N_HYPEROPT_PROBES = 10
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
    colorama.init()
    # алгоритм сэмплирования гиперпараметров
    HYPEROPT_ALGO = tpe.suggest
    space = {
        'iterations': 5,
        'depth': hp.quniform("depth", 3, 15, 1),
        'rsm': hp.uniform('rsm', 0.75, 1.0),
        'learning_rate': 0.1,
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
        'random_strength': hp.quniform("random_strenght", 3, 12, 1),
        'bagging_temperature': hp.quniform("bagging_temperature", 0, 10, 1),
    }
    trials = Trials()
    best = hyperopt.fmin(fn=objective(space, folds, test, X, Y, X_test, cat_features),
                         space=space,
                         algo=HYPEROPT_ALGO,
                         max_evals=N_HYPEROPT_PROBES,
                         trials=trials,
                         verbose=1)

    print('-' * 50)
    print('The best params:')
    print(best)
    print('\n\n')
    best.update({'iterations': 50000, 'learning_rate': 0.1, 'random_seed': 42, 'loss_function': 'RMSE'})
    params = best
else:
    params = {'depth': 15, 'iterations': 100000, 'learning_rate': 0.1, 'random_seed': 42, 'loss_function': 'RMSE'}

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=False, random_state=246)
model = CatBoostRegressor(**params)

prediction = np.zeros(test.shape[0])
rmse_list = []

for fold_n, (train_index, test_index) in enumerate(folds.split(X.atm_group)):
    print('Fold:', fold_n)
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]

    model.fit(X_train, Y_train, cat_features=cat_features,
              eval_set=[(X_valid, Y_valid)],
              early_stopping_rounds=50, verbose=100)

    y_pred = model.predict(X_test, ntree_end=model.best_iteration_)
    prediction += y_pred
    rmse_fold = rmse(Y_valid, model.predict(X_valid, ntree_end=model.best_iteration_))
    rmse_list.append(rmse_fold)

prediction /= n_fold

print('Final RMSE:{}'.format(sum(rmse_list)/len(rmse_list)))

submission = pd.DataFrame(prediction, index=test.index, columns=['target'])
submission.reset_index(level=0, inplace=True)
submission.index.name = ""
submission.to_csv('catboost_submit.csv',index=False)