import pandas as pd
from pathlib import Path

from . import support

load_parameters={
    'iris':
    {
        'no':1,
        'type':'classifier',
        'filename':'iris.data',
        'sep':',',
        'header':None,
        'index_col':None
    },
    'sonar':
    {
        'no':2,
        'type':'classifier',
        'filename':'sonar.all-data',
        'sep':',',
        'header':None,
        'index_col':None
    },
    'glass':
    {
        'no':3,
        'type':'classifier',
        'filename':'glass.data',
        'sep':',',
        'header':None,
        'index_col':0
    },
    'airfoil_self_noise':
    {
        'no':4,
        'type':'regressor',
        'filename':'airfoil_self_noise.dat',
        'sep':r'\t',
        'header':None,
        'index_col':None
    },
    'winequality-red':
    {
        'no':5,
        'type':'regressor',
        'filename':'winequality-red.csv',
        'sep':';',
        'header':0,
        'index_col':None
    },
    'winequality-white':
    {
        'no':6,
        'type':'regressor',
        'filename':'winequality-white.csv',
        'sep':';',
        'header':0,
        'index_col':None
    }
}

def load_data(name='iris'):
    # classifier_files=['iris.data','sonar.all-data','glass.data']
    # classifire_params=[(',',None,None),(',',None,None),(',',None,0)]
    # regressor_files=['airfoil_self_noise.dat','winequality-red.csv','winequality-white.csv']
    # regressor_params=[(r'\t',None,None),(';',0,None),(';',0,None)]

    current_para=load_parameters[name]
    path=Path(__file__).resolve().parent/'data'/current_para['filename']
    df = pd.read_csv( 
        path, 
        sep=current_para['sep'],
        header=current_para['header'],
        index_col=current_para['index_col'],
        engine='python')
    x=df[df.columns[:-1]].values

    if current_para['type'] == 'classifier':
        # ラベルを、ラベルの番号とそのラベルの属する可能性の配列で表現する。
        y,clz = support.clz_to_prob(df[df.columns[-1]])
        return x,y,clz
    
    else:
        y= df[df.columns[-1]].values.reshape( (-1,1) )
        return x,y