import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error

from . import load


def compute_f1_ac(plf, x, y):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    f1 = []
    pr = []
    n = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        plf.fit(x_train, y_train)
        z = plf.predict(x_test)
        z = z.argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        f1.append(f1_score(y_test, z, average='weighted'))
        pr.append(accuracy_score(y_test, z))
        n.append(len(x_test)/len(x))

    f1 = np.average(f1, weights=n)
    ac = np.average(pr, weights=n)

    return f1, ac


def compute_r2_mse(plf, x, y):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    r2 = []
    ma = []
    n = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        plf.fit(x_train, y_train)
        z = plf.predict(x_test)
        r2.append(r2_score(y_test, z))
        ma.append(mean_squared_error(y_test, z))
        n.append(len(x_test)/len(x))
    r2 = np.average(r2, weights=n)
    mse = np.average(ma, weights=n)

    return r2, mse


def evaluate_classifier(
    c_model, 
    targets=['iris', 'sonar','glass']):
    '''
    クラス分類器について評価する
    '''
    classifier_problems = []
    for name, para in load.load_parameters.items():
        if para['type'] == 'classifier':
            no = para['no']
            classifier_problems.append({
                'name': name,
                'no': no})

    result = pd.DataFrame(
        columns=['score_id', 'target', 'function', 'score'],
        index=range(len(classifier_problems)*2))

    nrow = 0

    for c_p in classifier_problems:
        name = c_p['name']
        # targetsの中にあれば、評価する。
        if name in targets:
            # データを読み込む
            x, y, clz = load.load_data(name)
            f1, ac = compute_f1_ac(c_model, x, y)
            result.loc[nrow, 'score'] = f1
            result.loc[nrow+1, 'score'] = ac
        else:
            result.loc[nrow, 'score'] = np.nan
            result.loc[nrow+1, 'score'] = np.nan

        result.loc[nrow, 'score_id'] = c_p['no']*2-1
        result.loc[nrow+1, 'score_id'] = c_p['no']*2
        result.loc[nrow, 'target'] = name
        result.loc[nrow+1, 'target'] = ' '
        result.loc[nrow, 'function'] = 'F1Score'
        result.loc[nrow+1, 'function'] = 'Accuracy'

        nrow += 2

    return result


def evaluate_regressor(r_model):
    '''
    回帰器について評価する
    '''
    regressor_problems = []
    for name, para in load.load_parameters.items():
        if para['type'] == 'regressor':
            no = para['no']
            regressor_problems.append({
                'name': name,
                'no': no})

    result = pd.DataFrame(
        columns=['score_id', 'target', 'function', 'score'],
        index=range(len(regressor_problems)*2))

    nrow = 0

    for r_p in regressor_problems:
        name = r_p['name']
        # データを読み込む
        x, y = load.load_data(name)

        result.loc[nrow, 'score_id'] = r_p['no']*2-1
        result.loc[nrow+1, 'score_id'] = r_p['no']*2
        result.loc[nrow, 'target'] = name
        result.loc[nrow+1, 'target'] = ' '
        result.loc[nrow, 'function'] = 'R2Score'
        result.loc[nrow+1, 'function'] = 'MeanSquared'

        r2, mse = compute_r2_mse(r_model, x, y)
        result.loc[nrow, 'score'] = r2
        result.loc[nrow+1, 'score'] = mse

        nrow += 2

    return result


def merge_result(*result_list):
    '''
    評価結果を結合する。
    '''
    data = []
    for name, paras in load.load_parameters.items():
        if paras['type'] == 'classifier':
            functions = ['F1Score', 'Accuracy']
        else:
            functions = ['R2Score', 'MeanSquared']
        row_data = []
        row_data.extend([paras['no']*2-1, name, functions[0]])
        data.append(row_data)
        row_data = []
        row_data.extend([paras['no']*2, ' ', functions[1]])
        data.append(row_data)

    result = pd.DataFrame(data, columns=['score_id', 'target', 'function'])
    for name_df in result_list:
        name = name_df[0]
        df = name_df[1].drop(columns=['target', 'function']
                             ).rename(columns={'score': name})
        result = result.merge(df, how='left', on='score_id')
    return result
