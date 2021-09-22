import numpy as np
import pandas as pd

from . import zeror, dstump, pruning


class WeightedZeroRule(zeror.ZeroRule):
    def fit(self, x, y, weight):
        # 重み付き平均を取る
        self.r = np.average(y, axis=0, weights=weight)
        return self


def w_gini(y, weight):
    i = y.argmax(axis=1)
    clz = set(i)
    score = 0.0
    for val in clz:
        p = weight[i == val].sum()
        score += p**2
    return 1.0 - score


def w_infogain(y, weight):
    i = y.argmax(axis=1)
    clz = set(i)
    score = 0.0
    for val in clz:
        p = weight[i == val].sum()
        if p != 0:
            score += p*np.log2(p)
    return -score


class WeightedDecisionStump(dstump.DescisionStump):
    def __init__(
            self,
            metric=w_infogain,
            leaf=WeightedZeroRule):
        """
        weightはfitメソッドの引数で、
        学習時、fit呼び出し直後にself.weightに格納される
        """
        super().__init__(metric=metric, leaf=leaf)
        self.weight = None

    def make_loss(self, y1, y2, l, r):
        """
        yをy1とy2で分割したときのMetrics関数の重み付き合計を返す。
        """
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        # Metric関数に渡す左右のデータの重み
        w1 = self.weight[l]/np.sum(self.weight[l])  # 重みの正規化
        w2 = self.weight[r]/np.sum(self.weight[r])  # 重みの正規化
        total = y1.shape[0]+y2.shape[0]
        m1 = self.metric(y1, w1)*(y1.shape[0]/total)
        m2 = self.metric(y2, w2)*(y2.shape[0]/total)
        return m1+m2

    def fit(self, x, y, weight):
        # 左右の葉を作成する
        self.weight = weight  # 重みを保存
        self.left = self.leaf()
        self.right = self.leaf()
        # データを左右の葉に振り分ける
        left, right = self.split_tree(x, y)
        # 重みをつけて左右の葉を学習させる
        if len(left) > 0:
            self.left.fit(
                x[left],
                y[left],
                weight[left]/np.sum(weight[left]))  # 重みの正規化

        if len(right) > 0:
            self.right.fit(
                x[right],
                y[right],
                weight[right]/np.sum(weight[right]))  # 重みの正規化

        return self

class WeightedDecistionTree(pruning.PrunedTree, WeightedDecisionStump):
    def __init__(
            self,
            max_depth=5,
            metric=w_gini,
            leaf=WeightedZeroRule,
            depth=1):
        super().__init__(
            max_depth=max_depth,
            metric=metric,
            leaf=leaf,
            depth=depth)
        self.weight = None

    def fit(self, x, y, weight):
        self.weight = weight  # 重みの保存
        # 深さ=1、根のノードのときのみ
        if self.depth == 1 and self.prunfnc is not None:
            # プルーニングに使うデータ
            x_t, y_t = x, y

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
                self.left.fit(
                    x[left],
                    y[left],
                    weight[left]/np.sum(weight[left]))
            if len(right) > 0:
                self.right = self.get_node()
                self.right.fit(
                    x[right],
                    y[right],
                    weight[right]/np.sum(weight[right]))
        # Critical Valueプルーニングでない場合、深さが最大深さに達していれば
        # 左右の葉を学習させる
        # (逆にCVプルーニングではこの段階で末端の葉の学習はしない)
        elif self.prunfnc != 'critical':
            if len(left) > 0:
                self.left.fit(
                    x[left],
                    y[left],
                    weight[left]/np.sum(weight[left]))
            if len(right) > 0:
                self.right.fit(
                    x[right],
                    y[right],
                    weight[right]/np.sum(weight[right]))

        # 深さ=1、根のノードの時のみ
        if self.depth == 1 and self.prunfnc is not None:
            if self.prunfnc == 'critical':
                #　学習スコアのMetrics関数のスコアを取得する
                score = []
                pruning.getscore(self, score)
                # スコアから残す枝の最大スコアを計算
                i = int(round(len(score)*self.critical))
                score_max = sorted(score)[min(i, len(score)-1)]
                # プルーニングを行う
                pruning.criticalscore(self, score_max)
                # 末端の葉を学習させる
                self.fit_leaf(x, y, weight)

        return self

    def fit_leaf(self, x, y, weight):
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)

        # 葉のみを学習させる
        if len(l) > 0:
            if isinstance(self.left, self.__class__):
                self.left.fit_leaf(x[l], y[l], weight[l])
            else:
                self.left.fit(x[l], y[l], weight[l])
        if len(r) > 0:
            if isinstance(self.right, self.__class__):
                self.right.fit_leaf(x[r], y[r], weight[r])
            else:
                self.right.fit(x[r], y[r], weight[r])
