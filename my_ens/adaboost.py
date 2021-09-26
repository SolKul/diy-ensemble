import numpy as np
from . import support, entropy, zeror, weighted


class AdaBoost:
    def __init__(
            self,
            boost=5,
            max_depth=5):

        self.boost = boost
        self.max_depth = max_depth
        self.trees = None
        self.alpha = None

    def fit(self, x, y):
        # ブースティングで使用する変数
        self.trees = []  # 各機械学習モデルの配列
        self.alpha = np.zeros((self.boost,))  # 貢献度の配列
        n_clz = y.shape[1]
        if n_clz != 2:
            return  # 基本のAdaBoostは2クラス分類のみ
        y_bin = y.argmax(axis=1)*2-1  # 1と-1の配列にする
        # 学習データに対する重み
        weights = np.ones((len(x),)) / len(x)
        # ブースティング
        for i in range(self.boost):
            # ここで決定木モデルを作成し、学習データに対して実行する
            # 決定木モデルを作成
            tree = weighted.WeightedDecisionTree(
                max_depth=self.max_depth,
                metric=weighted.w_gini,
                leaf=weighted.WeightedZeroRule)
            tree.fit(x, y, weights)
            # 一度学習データに対して予測を実行する
            z = tree.predict(x)
            z_bin = z.argmax(axis=1)*2 - 1  # 1と-1の配列にする
            # 正解したデータを探す
            filtered = (z_bin == y_bin)
            err = weights[filtered == False].sum()  # 不正解の位置にある重みの合計
            print(f'itre #{i} -- error = {err:.3f}')

            # ここで早期終了の条件をチェックする
            # 終了条件
            if i == 0 and err == 0:  # 最初に完全に学習してしまった
                self.treees.append(tree)  # 最初のモデルだけ
                self.alpha = self.alpha[:i+1]
                break
            if err > 0.5 or err == 0:  # 正解率が0.5を下回ったか、全部正解か。
                self.apha = self.alpha[:i]  # 1つ前まで
                break

            # ここでAdaBoostの計算を行う
            # 学習したモデルを追加
            self.trees.append(tree)
            # AdaBoostの計算
            self.alpha[i] = np.log((1.0-err)/err)/2.0  # 式9
            weights *= np.exp(-1.0*self.alpha[i]*y_bin*z_bin)  # 式10
            weights /= weights.sum()  # 重みの正規化

    def predict(self, x):
        # 各モデルの出力の合計
        z = np.zeros((len(x)))
        for i, tree in enumerate(self.trees):
            p = tree.predict(x)
            p_bin = p.argmax(axis=1)*2-1 # 1と-1の配列にする
            z += p_bin*self.alpha[i] # 貢献度を加味して足していく
        # 合計した出力を、その符号で[0,1]と[1,0]の配列にする。
        return np.array([z <= 0, z > 0]).astype(int).T

    def __str__(self):
        s = []
        for i , t in enumerate(self.trees):
            s.append(f'tree: #{i+1} -- weight={self.alpha[i]}')
            s.append(str(t))
        return '\n'.join(s)

if __name__ == '__main__':

    import pandas as pd
    from pathlib import Path
    from . import evaluate

    # 引数の辞書
    evaluate_list = [
        {
            'name': 'AdaBoost',
            'classifier': {
                'class': AdaBoost,
                'args': {}
            }
        }
    ]
    result_dict = {}
    for eval_para in evaluate_list:

        if 'classifier' in eval_para:
            # 引数展開を利用して引数を渡す。
            c_class = eval_para['classifier']['class']
            c_model = c_class(**eval_para['classifier']['args'])
            c_result = evaluate.evaluate_classifier(
                c_model,
                targets=['sonar'])
        else:
            c_result=pd.DataFrame()

        if 'regressor' in eval_para:
            r_class = eval_para['regressor']['class']
            r_model = r_class(**eval_para['classifier']['args'])
            r_result = evaluate.evaluate_regressor(r_model)
        else:
            r_result=pd.DataFrame()

        # 結果のDataFrameを上下で結合して辞書に格納
        result_dict[eval_para['name']] = pd.concat(
            [c_result, r_result], axis=0)

    result_list = []
    for name, df in result_dict.items():
        result_list.append((name, df))

    result = evaluate.merge_result(*result_list)

    print(result)
    result_path = Path(__file__).resolve().parent / \
        'processed_data'/'adaboost.csv'
    result.to_csv(result_path, index=None)
