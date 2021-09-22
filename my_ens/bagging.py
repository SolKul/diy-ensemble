import numpy as np
import pandas as pd
import random

from . import zeror, entropy, dtree, pruning


class Bagging:
    def __init__(
            self,
            n_trees=5,
            ratio=1.0,
            tree=pruning.PrunedTree,
            tree_params={}):

        self.n_trees = n_trees
        self.ratio = ratio
        self.tree=tree
        self.tree_params = tree_params
        self.trees = []

    def fit(self, x, y):
        # 全ての木をいったん削除
        self.trees.clear()
        # 機械学習モデル用のデータの数
        n_sample = int(round(len(x)*self.ratio))
        for _ in range(self.n_trees):
            # 重複ありランダムサンプルで学習データのインデックスを生成する
            index = np.random.choice(np.arange(x.shape[0]), size=n_sample)
            # 新しい機械学習モデルを作成する
            tree = self.tree(**self.tree_params)
            # 機械学習モデルを1つ学習させる
            tree.fit(x[index], y[index])
            # 機械学習モデルを保存する
            self.trees.append(tree)
        return self

    def predict(self, x):
        # すべての機械学習モデルの結果をリストにする
        z = [tree.predict(x) for tree in self.trees]
        # リスト内の結果の平均をとって返す
        return np.mean(z, axis=0)

    def __str__(self):
        text_list = [f'tree#{i}\n{tree}' for i, tree in enumerate(self.trees)]
        return '\n'.join(text_list)

if __name__ == '__main__':

    import pandas as pd
    from pathlib import Path
    from . import evaluate

    # 引数の辞書
    args_list = [
        {'name': '5 trees',
         'classifier':
            {'n_trees': 5,
             'tree_params': 
                {'metric':entropy.gini}
            },
         'regresser':
            {'n_trees': 5,
             'tree_params': 
                {'metric':entropy.deviation}
            }
        },
        {'name': '10 trees',
         'classifier':
            {'n_trees': 10,
             'tree_params': 
                {'metric':entropy.gini}
            },
         'regresser':
            {'n_trees': 10,
             'tree_params': 
                {'metric':entropy.deviation}
            }
        },
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
    for args_dict in args_list:

        # 引数展開を利用して引数を渡す。
        c_model = Bagging(**args_dict['classifier'])
        c_result = evaluate.evaluate_classifier(c_model)

        r_model = Bagging(**args_dict['regresser'])
        r_result = evaluate.evaluate_regressor(r_model)

        # 結果のDataFrameを上下で結合して辞書に格納
        result_dict[args_dict['name']] = pd.concat(
            [c_result, r_result], axis=0)

    result_list = []
    for name, df in result_dict.items():
        result_list.append((name, df))

    result = evaluate.merge_result(*result_list)

    print(result)
    result_path = Path(__file__).resolve().parent / \
        'processed_data'/'bagging.csv'
    result.to_csv(result_path, index=None)
