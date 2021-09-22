import re
import numpy as np
import pandas as pd
from sklearn import model_selection
from pathlib import Path

from . import support

from sklearn.model_selection import KFold,cross_validate
from sklearn.svm import SVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor

def evaluate_base():
    models=[
        ('SVM',SVC(random_state=1),SVR()),
        ('GaussianProcess',GaussianProcessClassifier(random_state=1),GaussianProcessRegressor(normalize_y=True,alpha=1,random_state=1)),
        ('KNeighbors',KNeighborsClassifier(),KNeighborsRegressor()),
        ('MLP',MLPClassifier(random_state=1),MLPRegressor(hidden_layer_sizes=(5),solver='lbfgs',random_state=1)),
    ]

    classifier_files=['iris.data','sonar.all-data','glass.data']
    classifire_params=[(',',None,None),(',',None,None),(',',None,0)]
    regressor_files=['airfoil_self_noise.dat','winequality-red.csv','winequality-white.csv']
    regressor_params=[(r'\t',None,None),(';',0,None),(';',0,None)]

    result = pd.DataFrame(
        columns=['target','function']+[m[0] for m in models],
        index=range(len(classifier_files+regressor_files)*2) )

    nrow=0
    for i,(c,p) in enumerate(zip(classifier_files,classifire_params)):
        path=Path(__file__).resolve().parent/'data'/c

        #　ファイルを読み込む
        df = pd.read_csv( path, sep=p[0],header=p[1],index_col=p[2])
        x=df[df.columns[:-1]].values
        # ラベルを、ラベルの番号とそのラベルの属する可能性の配列で表現する。
        y,clz = support.clz_to_prob(df[df.columns[-1]])

        # 結果の表にファイル名からデータセットの種類と、評価関数用の行を作る。
        result.loc[nrow,'target']=re.split(r'[._]',c)[0]
        result.loc[nrow+1, 'target']=' '
        result.loc[nrow,'function']='F1Score'
        result.loc[nrow+1,'function']='Accuracy'

        #すべてのアルゴリズムで評価する
        for l, c_m, r_m in models:
            # Scikit-learnの関数で効果検証した結果のスコアを取得する。
            kf=KFold(n_splits=5,random_state=1,shuffle=True)
            s=cross_validate(c_m,x,y.argmax(axis=1),cv=kf,scoring=('f1_weighted','accuracy'))
            result.loc[nrow,l]=np.mean(s['test_f1_weighted'])
            result.loc[nrow+1,l]=np.mean(s['test_accuracy'])

        nrow+=2

    # 次に回帰アルゴリズムを評価する
    for i,(c,p) in enumerate(zip(regressor_files,regressor_params)):
        path=Path(__file__).resolve().parent/'data'/c

        #　ファイルを読み込む
        df = pd.read_csv( path, sep=p[0],header=p[1],index_col=p[2])
        x= df[df.columns[:-1]].values
        y= df[df.columns[-1]].values.reshape( (-1,) )

        # 結果の表にファイル名からデータセットの種類と、評価関数用の行を作る。
        result.loc[nrow,'target']=re.split(r'[._]',c)[0]
        result.loc[nrow+1, 'target']=' '
        result.loc[nrow,'function']='R2Score'
        result.loc[nrow+1,'function']='MeanSquared'

        #すべてのアルゴリズムで評価する
        for l, c_m, r_m in models:
            # Scikit-learnの関数で効果検証した結果のスコアを取得する。
            kf=KFold(n_splits=5,random_state=1,shuffle=True)
            s=cross_validate(r_m,x,y,cv=kf,scoring=('r2','neg_mean_squared_error'))
            result.loc[nrow,l]=np.mean(s['test_r2'])
            result.loc[nrow+1,l]=-np.mean(s['test_neg_mean_squared_error'])

        nrow+=2

    return result

if __name__ == '__main__':
    result=evaluate_base()

    # 結果を保存
    print( result )
    p_path=Path(__file__).resolve().parent/'processed_data'/'baseline.csv'
    result.to_csv(p_path,index=None)