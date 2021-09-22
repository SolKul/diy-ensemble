import pandas as pd
from pathlib import Path

from pandas.core.indexes import base

from . import myresult

def compare():
    base_path=Path(__file__).resolve().parent/'processed_data'/'baseline.csv'
    base_result=pd.read_csv(base_path)
    my_result=myresult.evaluate_my()


    compare_result=pd.concat([my_result,base_result.iloc[:,2:]],axis=1)
    p_path=Path(__file__).resolve().parent/'processed_data'/'compare.csv'
    compare_result.to_csv(p_path,index=None)

    return compare_result

def concat_result():
    base_path=Path(__file__).resolve().parent/'processed_data'/'baseline.csv'
    base_result=pd.read_csv(base_path)
    my_path=Path(__file__).resolve().parent/'processed_data'/'dstump.csv'
    my_result=pd.read_csv(my_path)
    my_result=my_result.rename(columns={'score':'dstump'})

    concat_result=pd.concat([my_result,base_result.iloc[:,2:]],axis=1)
    p_path=Path(__file__).resolve().parent/'processed_data'/'concat.csv'
    concat_result.to_csv(p_path,index=None)

    return concat_result

if __name__ == '__main__':
    compare_result=compare()

    print(compare_result)
    p_path=Path(__file__).resolve().parent/'processed_data'/'compare.csv'
    compare_result.to_csv(p_path,index=None)