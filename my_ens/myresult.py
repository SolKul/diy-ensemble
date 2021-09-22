import pandas as pd
from pathlib import Path
# import re

from . import load,support,evaluate,zeror,linear

def evaluate_my():
    classifier_problems=['iris','sonar','glass']
    regressor_problems=['airfoil_self_noise','winequality-red','winequality-white']
    # classifier_files=['iris.data','sonar.all-data','glass.data']
    # classifire_params=[(',',None,None),(',',None,None),(',',None,0)]
    # regressor_files=['airfoil_self_noise.dat','winequality-red.csv','winequality-white.csv']
    # regressor_params=[(r'\t',None,None),(';',0,None),(';',0,None)]

    models=[
        ("ZeroRule",zeror.ZeroRule(),zeror.ZeroRule()),
        ("Linear",zeror.ZeroRule(),linear.Linear()),
        ("LinearNE",zeror.ZeroRule(),linear.LinearNE())
    ]

    result = pd.DataFrame(
        columns=['target','function']+[m[0] for m in models],
        index=range(len(classifier_problems+regressor_problems)*2) )

    nrow=0

    for i,name in enumerate(classifier_problems):
        # データを読み込む
        x,y,clz=load.load_data(name)

        result.loc[nrow,'target']=name
        result.loc[nrow+1, 'target']=' '
        result.loc[nrow,'function']='F1Score'
        result.loc[nrow+1,'function']='Accuracy'

        for l, c_m, r_m in models:
            f1,ac=evaluate.compute_f1_ac(c_m,x,y)
            result.loc[nrow,l]=f1
            result.loc[nrow+1,l]=ac

        nrow += 2

    for i,name in enumerate(regressor_problems):
        # データを読み込む
        x,y=load.load_data(name)

        result.loc[nrow,'target']=name
        result.loc[nrow+1, 'target']=' '
        result.loc[nrow,'function']='R2Score'
        result.loc[nrow+1,'function']='MeanSquared'

        for l, c_m, r_m in models:
            r2,mse=evaluate.compute_r2_mse(r_m,x,y)
            result.loc[nrow,l]=r2
            result.loc[nrow+1,l]=mse

        nrow += 2

    return result

if __name__ == '__main__':
    result=evaluate_my()

    print(result)
    p_path=Path(__file__).resolve().parent/'processed_data'/'myresult.csv'
    result.to_csv(p_path,index=None)