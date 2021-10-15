from my_ens.zeror import ZeroRule
import numpy as np

from . import support, entropy, linear, dstump


class DecisionTree(dstump.DescisionStump):
    def __init__(
            self,
            max_depth=5,
            metric=entropy.gini,
            leaf=ZeroRule,
            depth=1):

        super().__init__(metric=metric, leaf=leaf)
        self.max_depth = max_depth
        self.depth = depth
        self.dimensions = 0

    def fit(self, x, y):
        '''
        左右の葉を作成する
        '''
        self.dimensions = x.shape[1]
        self.left = self.leaf()
        self.right = self.leaf()

        # データを左右に分割する
        left, right = self.split_tree(x, y)
        # 現在のノードの深さが最大深さに達していないなら、
        if self.depth < self.max_depth:
            # 実際にデータがあるなら、DescitionTreeクラスで置き換える。
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
        # 左右の葉を学習させる
        if len(left) > 0:
            self.left.fit(x[left], y[left])
        if len(right) > 0:
            self.right.fit(x[right], y[right])

        return self

    def get_node(self):
        '''
        新しくノードを作成する。
        '''
        return self.__class__(
            max_depth=self.max_depth,
            metric=self.metric,
            leaf=self.leaf,
            depth=self.depth+1)

    def split_tree_fast(self, x, y):
        '''
        データを分割して左右の枝に属するインデックスを返す
        '''

        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        # データの前準備
        ytil = y[:, np.newaxis]
        xindex = np.argsort(x, axis=0)
        ysot = np.take(ytil, xindex, axis=0)
        for f in range(x.shape[0]):
            # 小さいほうからfの位置にある値で分割
            l = xindex[:f, :]
            r = xindex[f:, :]
            ly = ysot[:f, :, 0, :]
            ry = ysot[f:, :, 0, :]

            # 全ての説明変数のスコアを求める
            loss = [self.make_loss(
                ly[:, yp, :],
                ry[:, yp, :],
                l[:, yp],
                r[:, yp])
                # 説明変数が１つ前の値と同じだったら、inf
                if x[xindex[f-1, yp], yp] != x[xindex[f, yp], yp] else np.inf
                for yp in range(x.shape[1])]

            # 最小のスコアになる次元
            i = np.argmin(loss)
            if score > loss[i]:
                score = loss[i]
                self.feat_index = i
                self.feat_val = x[xindex[f, i], i]

        # 実際に分割するインデックスを取得
        x_filter = x[:, self.feat_index] < self.feat_val
        left = np.where(x_filter)[0].tolist()
        right = np.where(x_filter == False)[0].tolist()
        self.score = score
        return left, right

    def split_tree(self, x, y):
        return self.split_tree_fast(x, y)

    def print_leaf(self, node, d=0):
        if isinstance(node, self.__class__):
            return '\n'.join([
                f"  {'+'*d}if feat[ {node.feat_index} ] <= {node.feat_val:.1f} then:",
                self.print_leaf(node.left, d+1),
                f"  {'|'*d}else:",
                self.print_leaf(node.right, d+1)])
        else:
            return f"  {'|'*(d-1)} {node}"

    def __str__(self):
        return self.print_leaf(self)

    def calc_feat_imp(self):
        """
        特徴量重要度を算出する
        """
        feat_imp_scores = np.zeros(self.dimensions)
        self.calc_rec_imp(feat_imp_scores)
        # 足して1になるように正規化
        return feat_imp_scores/np.sum(feat_imp_scores)

    def calc_rec_imp(self, feat_imp_scores):
        """
        再起関数で特徴量の重要度を算出する関数
        """
        # まず、同じ深さの木全体の合計スコアを算出する
        # layer_score=2**(dt.max_depth-dt.depth)
        # それを同じ深さの木の数で割る。
        # my_score=layer_score/(2**(dt.depth-1))
        # これは次の式と同じ
        my_score = 2**(1+self.max_depth-2*self.depth)
        # 分割に選ばれた特徴量の重要度スコアに足し合わせる。
        feat_imp_scores[self.feat_index] += my_score
        # もし左がDecisionTreeのインスタンスなら、
        if isinstance(self.left, DecisionTree):
            self.left.calc_rec_imp(feat_imp_scores)
        # もし右がDecisionTreeのインスタンスなら、
        if isinstance(self.right, DecisionTree):
            self.right.calc_rec_imp(feat_imp_scores)
        return feat_imp_scores


'''
split_tree_fast説明

>> ysot[0,:,:]
array([[[1., 0., 0.]],# 1番目の説明変数が１番小さいときのyのクラス
       [[0., 1., 0.]],# 2番目の説明変数が１番小さいときのyのクラス
       [[1., 0., 0.]],
       [[1., 0., 0.]]])

>>ly = ysot[:2,:,0,:] # 0次元目のいくつか、1次元目を全部、2次元目は0を指定、3次元目は全部
array([[[1., 0., 0.],# 1番目の説明変数が1番目に小さい数のときのyのクラス
        [0., 1., 0.],# 2番目の説明変数が1番目に小さい数のときのyのクラス
        [1., 0., 0.],
        [1., 0., 0.]],

       [[1., 0., 0.],#1番目の説明変数が2番目に小さい数のときのyのクラス
        [0., 1., 0.],#2番目の説明変数が2番目に小さい数のときのyのクラス
        [1., 0., 0.],
        [1., 0., 0.]]])

>>ly[:,0,:]# 1番目の説明変数が1番と2番目に小さい数のときのyのクラスを取り出す
array([[1., 0., 0.], #１番目の説明変数が1番目に小さい数のときのyのクラス
       [1., 0., 0.]]) #１番目の説明変数が2番目に小さい数のときのyのクラス
'''


class DecisionTree2(dstump.DescisionStump2):
    def __init__(
            self,
            max_depth=5,
            metric=entropy.gini,
            leaf=ZeroRule,
            depth=1):

        super().__init__(metric=metric, leaf=leaf)
        self.max_depth = max_depth
        self.depth = depth

    def fit(self, x, y):
        '''
        左右の葉を作成する
        '''
        self.left = self.leaf()
        self.right = self.leaf()

        # データを左右に分割する
        left, right = self.split_tree(x, y)
        # 現在のノードの深さが最大深さに達していないなら、
        if self.depth < self.max_depth:
            # 実際にデータがあるなら、DescitionTreeクラスで置き換える。
            if np.any(left):
                self.left = self.get_node()
            if np.any(right):
                self.right = self.get_node()
        # 左右の葉を学習させる
        if np.any(left):
            self.left.fit(x[left, :], y[left, :])
        if np.any(right):
            self.right.fit(x[right, :], y[right, :])

        return self

    def get_node(self):
        '''
        新しくノードを作成する。
        '''
        return self.__class__(
            max_depth=self.max_depth,
            metric=self.metric,
            leaf=self.leaf,
            depth=self.depth+1)


if __name__ == '__main__':

    import pandas as pd
    from pathlib import Path
    from . import evaluate

    result_dict = {}
    for depth in [3, 5, 7]:
        c_model = DecisionTree(max_depth=depth)
        c_result = evaluate.evaluate_classifier(c_model)

        r_model = DecisionTree(
            max_depth=depth,
            metric=entropy.deviation)
        r_result = evaluate.evaluate_regressor(r_model)

        result_dict["{}d DTree".format(depth)] = pd.concat(
            [c_result, r_result], axis=0)

    result_list = []
    for name, df in result_dict.items():
        result_list.append((name, df))

    result = evaluate.merge_result(*result_list)

    print(result)
    result_path = Path(__file__).resolve().parent/'processed_data'/'dtree.csv'
    result.to_csv(result_path, index=None)
