import numpy as np

from . import support,evaluate

class ZeroRule:
    def __init__(self):
        self.r=None

    def fit(self, x, y):
        self.r = np.mean( y, axis=0)
        return self

    def predict(self, x):
        z = np.zeros( (len(x),self.r.shape[0]) )
        return z + self.r

    def __str__(self):
        return str(self.r)

if __name__ == '__main__':
    # import pandas as pd
    # ps = support.get_base_args()
    # args = ps.parse_args()

    # df=pd.read_csv(
    #     args.input,
    #     sep=args.separator,
    #     header=args.header,
    #     index_col=args.indexcol)
    # x=df[df.columns[:-1]].values

    # if not args.regression:
    #     y,clz=support.clz_to_prob( df[df.columns[-1]] )
    #     plf=ZeroRule()
    #     support.report_classifier(plf,x,y,clz,args.crossvalidate)
    # else:
    #     y = df[df.columns[-1]].values.reshape((-1,1))   
    #     plf=ZeroRule()
    #     support.report_regressor(plf,x,y,args.crossvalidate)

    import pandas as pd
    from pathlib import Path

    cz=ZeroRule()
    cz_result=evaluate.evaluate_classifier(cz)

    rz=ZeroRule()
    rz_result=evaluate.evaluate_regressor(rz)

    zero_result=pd.concat([cz_result,rz_result],axis=0)

    result_list=[
        ('ZeroRule',zero_result)
    ]
    result=evaluate.merge_result(*result_list)

    print(result)
    result_path=Path(__file__).resolve().parent/'processed_data'/'zeror.csv'
    result.to_csv(result_path,index=None)