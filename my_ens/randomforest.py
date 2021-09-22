import numpy as np
import pandas as pd
import random

from . import zeror, entropy, pruning


class RandomTree(pruning.PrunedTree):
    def __init__(
            self,
            features=5,
            max_depth=5,
            metric=entropy.gini,
            leaf=zeror.ZeroRule,
            depth=1):

        super().__init__(
            max_depth=max_depth,
            metric=metric,
            leaf=leaf,
            depth=depth)
        self.features = features

    def split_tree(self, x, y):
        # 説明変数の次元から、ランダムに使用する次元を選択する。
        # (ランダムな数番目な説明変数をfeatures個選択する)
        index = np.random.choice(
            np.arange(x.shape[1]),
            size=min(self.features, x.shape[1]),
            replace=False)
        # 説明変数の選択された次元のみ使用して分割
        result = super().split_tree(x[:, index], y)
        # 分割の次元を元の次元に戻す
        self.feat_index = index[self.feat_index]
        return result

    def get_node(self):
        # 新しくノードを作成する
        return self.__class__(
            features=self.features,
            max_depth=self.max_depth,
            metric=self.metric,
            leaf=self.leaf,
            depth=self.depth+1)


if __name__ == '__main__':

    import pandas as pd
    from pathlib import Path
    from . import bagging, evaluate

    # 引数の辞書
    evaluate_list = [
        {
            'name': '5 trees',
            'classifier': {
                'class': bagging.Bagging,
                'args': {
                    'n_trees': 5,
                    'tree': RandomTree,
                    'tree_params': {'metric': entropy.gini}
                }
            },
            'regresser': {
                'class': bagging.Bagging,
                'args': {
                    'n_trees': 5,
                    'tree': RandomTree,
                    'tree_params': {'metric': entropy.deviation}
                }
            }
        },
        {
            'name': '10 trees',
            'classifier': {
                'class': bagging.Bagging,
                'args': {
                    'n_trees': 10,
                    'tree': RandomTree,
                    'tree_params': {'metric': entropy.gini}
                }
            },
            'regresser': {
                'class': bagging.Bagging,
                'args': {
                    'n_trees': 10,
                    'tree': RandomTree,
                    'tree_params': {'metric': entropy.deviation}
                }
            }
        }
        # {'name': '20 trees',
        #  'classifier':
        #     {'n_trees': 20,
        #      'tree_params':
        #         {'metric':entropy.gini}
        #     },
        #  'regresser':
        #     {'n_trees': 20,
        #      'tree_params':
        #         {'metric':entropy.deviation}
        #     }
        # }
    ]
    result_dict = {}
    for eval_para in evaluate_list:

        # 引数展開を利用して引数を渡す。
        c_class = eval_para['classifier']['class']
        c_model = c_class(**eval_para['classifier']['args'])
        c_result = evaluate.evaluate_classifier(c_model)

        r_class = eval_para['regresser']['class']
        r_model = r_class(**eval_para['classifier']['args'])
        r_result = evaluate.evaluate_regressor(r_model)

        # 結果のDataFrameを上下で結合して辞書に格納
        result_dict[eval_para['name']] = pd.concat(
            [c_result, r_result], axis=0)

    result_list = []
    for name, df in result_dict.items():
        result_list.append((name, df))

    result = evaluate.merge_result(*result_list)

    print(result)
    result_path = Path(__file__).resolve().parent / \
        'processed_data'/'random_forest.csv'
    result.to_csv(result_path, index=None)
