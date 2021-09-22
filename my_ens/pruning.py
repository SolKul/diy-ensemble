import numpy as np
import pandas as pd

from . import zeror, entropy, dtree


class PrunedTree(dtree.DecisionTree):
    def __init__(
            self,
            prunfnc='critical',
            pruntest=False,
            splitratio=0.5,
            critical=0.8,
            max_depth=5,
            metric=entropy.gini,
            leaf=zeror.ZeroRule,
            depth=1):

        super().__init__(
            max_depth=max_depth,
            metric=metric,
            leaf=leaf,
            depth=depth)
        self.prunfnc = prunfnc  # プルーニング用関数
        self.pruntest = pruntest  # プルーニング用にテスト用データを取り分けるか
        self.splitratio = splitratio  # プルーニング用テストデータの割合
        self.critical = critical  # 'critical'プルーニング用の閾値

    def get_node(self):
        #　新しくノードを作成する
        return self.__class__(
            prunfnc=self.prunfnc,
            pruntest=self.pruntest,
            splitratio=self.splitratio,
            critical=self.critical,
            max_depth=self.max_depth,
            metric=self.metric,
            leaf=self.leaf,
            depth=self.depth+1)

    def fit(self, x, y):
        # 深さ=1、根のノードのときのみ
        if self.depth == 1 and self.prunfnc is not None:
            # プルーニングに使うデータ
            x_t, y_t = x, y
            # プルーニング用にテスト用データを取り分けるなら
            if self.pruntest:
                # 学習データとテスト用データを別にする
                n_test = int(round(len(x)*self.splitratio))
                n_idx = np.random.permutation(len(x))
                tmpx = x[n_idx[n_test:]]
                tmpy = y[n_idx[n_test:]]
                x_t = x[n_idx[:n_test]]
                y_t = y[n_idx[:n_test]]
                x = tmpx
                y = tmpy

        # 決定木の学習・・・"critical"プルーニング時は木の分割のみ
        self.left = self.leaf()
        self.right = self.leaf()
        left, right = self.split_tree(x, y)

        # 現在のノードの深さが最大深さに達していないなら、
        if self.depth < self.max_depth:
            # len(left)>0のもののみTreeに置き換えないと、
            # self.left.left、self.left.rightにNoneが生成されてしまう恐れがある
            if len(left) > 0:
                self.left = self.get_node()
                self.left.fit(x[left], y[left])
            if len(right) > 0:
                self.right = self.get_node()
                self.right.fit(x[right], y[right])
        # Critical Valueプルーニングでない場合、深さが最大深さに達していれば
        # 左右の葉を学習させる
        # (逆にCVプルーニングではこの段階で末端の葉の学習はしない)
        elif self.prunfnc != 'critical':
            if len(left) > 0:
                self.left.fit(x[left], y[left])
            if len(right) > 0:
                self.right.fit(x[right], y[right])

        # 深さ=1、根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == 'reduce':
                # プルーニングを行う
                reducederror(self, x_t, y_t)
            elif self.prunfnc == 'critical':
                #　学習スコアのMetrics関数のスコアを取得する
                score = []
                getscore(self, score)
                if len(score) > 0:
                    # スコアから残す最大スコア量を計算
                    i = int(round(len(score)*self.critical))
                    score_max = sorted(score)[min(i, len(score)-1)]
                    # プルーニングを行う
                    criticalscore(self, score_max)
                # 末端の葉を学習させる
                self.fit_leaf(x, y)

        return self

    def fit_leaf(self, x, y):
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)

        # 葉のみを学習させる
        if len(l) > 0:
            if isinstance(self.left, self.__class__):
                self.left.fit_leaf(x[l], y[l])
            else:
                self.left.fit(x[l], y[l])
        if len(r) > 0:
            if isinstance(self.right, self.__class__):
                self.right.fit_leaf(x[r], y[r])
            else:
                self.right.fit(x[r], y[r])


def reducederror(node, x, y):
    '''
    再帰関数で決定木の枝刈りをする
    '''
    # 1.決定木のノードかを判定
    if isinstance(node, PrunedTree):
        # 左右の分割を得る
        feat = x[:, node.feat_index]
        val = node.feat_val
        l, r = node.make_split(feat, val)
        # 2.左右どちらかに全てのデータが行く場合、
        if val is np.inf or len(r) == 0:
            # 3,4左のノードを枝刈りし、それを返す
            return reducederror(node.left, x, y)
        elif len(l) == 0:
            # 3,4右のノードを枝刈りし、それを返す
            return reducederror(node.right, x, y)

        # 5,6. 左の枝を枝刈りし、それをleftに代入
        node.left = reducederror(node.left, x[l], y[l])
        # 7,8. 右の枝を枝刈りし、それをrightに代入
        node.right = reducederror(node.right, x[r], y[r])

        # 9. 学習データに対するスコアを計算する
        p1 = node.predict(x)
        p2 = node.left.predict(x)
        p3 = node.right.predict(x)

        # クラス分類かどうか
        if y.shape[1] > 1:
            # 誤分類の個数をスコアにする
            ya = y.argmax(axis=1)
            d1 = np.sum(p1.argmax(axis=1) != ya)
            d2 = np.sum(p2.argmax(axis=1) != ya)
            d3 = np.sum(p3.argmax(axis=1) != ya)
        else:
            # 二乗平均誤差をスコアにする。
            d1 = np.mean((p1-y)**2)
            d2 = np.mean((p2-y)**2)
            d3 = np.mean((p3-y)**2)
        # 本来なら、親の予測は子の予測を組み合わせたものなので、
        # 親の予測のほうが高精度なはずのでd1<d2,d3となるはずだが、
        # どちらかの子だけで予測しても精度が親以上であれば、
        # d2,d3<=d1となる。すなわち、左右の枝どちらかだけでスコアが悪化しない。
        if d2 <= d1 or d3 <= d1:
            # スコアの良いほうの枝を返す。
            if d2 < d3:
                return node.left
            else:
                return node.right

    # 10.現在のノードを返す。
    return node


def getscore(node, score):
    # 1.決定木のノードかを判定
    if isinstance(node, PrunedTree):
        if node.score >= 0 and node.score is not np.inf:
            score.append(node.score)
        getscore(node.left, score)
        getscore(node.right, score)


def criticalscore(node, score_max):
    # 1.決定木のノードかを判定
    if isinstance(node, PrunedTree):
        # 2,3. 左の枝を枝刈りし、それをleftに代入
        node.left = criticalscore(node.left, score_max)
        # 4,5. 右の枝を枝刈りし、それをrightに代入
        node.right = criticalscore(node.right, score_max)
        # 6.もし閾値よりスコアが悪ければ
        if node.score > score_max:
            # 左が葉かどうか
            leftisleaf = not isinstance(node.left, PrunedTree)
            # 右が葉かどうか
            rightisleaf = not isinstance(node.right, PrunedTree)
            # 両方とも葉なら1つの葉にする
            if leftisleaf and rightisleaf:
                return node.left
            # どちらかが枝なら、枝のほうを残す
            elif leftisleaf:
                return node.right
            elif rightisleaf:
                return node.left
            # どちらも枝ならばスコアの良い方を残す。
            elif node.left.score < node.right.score:
                return node.left
            else:
                return node.right
    return node


if __name__ == '__main__':

    import pandas as pd
    from pathlib import Path
    from . import evaluate

    # 引数の辞書
    args_list = [
        {'name': 'reduce_same',
         'classifier':
            {'prunfnc': 'reduce',
             'pruntest': False,
             'splitratio': 0.5,
             'metric': entropy.gini},
         'regresser':
            {'prunfnc': 'reduce',
             'pruntest': False,
             'splitratio': 0.5,
             'metric': entropy.deviation}
         },
        {'name': 'reduce_0.5',
         'classifier':
            {'prunfnc': 'reduce',
             'pruntest': True,
             'splitratio': 0.5,
             'metric': entropy.gini},
         'regresser':
            {'prunfnc': 'reduce',
             'pruntest': True,
             'splitratio': 0.5,
             'metric': entropy.deviation}
         },
        {'name': 'critical',
         'classifier':
            {'prunfnc': 'critical',
             'pruntest': False,
             'splitratio': 0.5,
             'metric': entropy.gini},
         'regresser':
            {'prunfnc': 'critical',
             'pruntest': False,
             'splitratio': 0.5,
             'metric': entropy.deviation}
         }
    ]
    result_dict = {}
    for args_dict in args_list:

        # 引数展開を利用して引数を渡す。
        c_model = PrunedTree(**args_dict['classifier'])
        c_result = evaluate.evaluate_classifier(c_model)

        r_model = PrunedTree(**args_dict['regresser'])
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
        'processed_data'/'prunedtree.csv'
    result.to_csv(result_path, index=None)
