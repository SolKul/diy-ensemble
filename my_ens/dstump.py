import numpy as np
from . import support, entropy, zeror


class DescisionStump:
    def __init__(
            self,
            metric=entropy.gini,
            leaf=zeror.ZeroRule):

        self.metric = metric
        self.leaf = leaf
        self.left = None
        self.right = None
        self.feat_index = 0
        self.feat_val = np.nan
        self.score = np.nan

    def make_split(self, feat, val):
        '''
        featをvalより小さいものとval以上で分割するインデックスを返す。
        '''
        left, right = [], []

        for i, v in enumerate(feat):
            if v < val:
                left.append(i)
            else:
                right.append(i)
        return left, right

    def make_loss(self, y1, y2, l, r):
        '''
        yをy1とy2で分割したときのMetrics関数の重み付き合計を返す。
        '''
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        total = y1.shape[0]+y2.shape[0]
        m1 = self.metric(y1)*(y1.shape[0]/total)
        m2 = self.metric(y2)*(y2.shape[0]/total)
        return m1 + m2

    def split_tree(self, x, y):
        '''
        データを分割して左右の枝に属するインデックスを返す
        '''
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        # 左右のインデックス
        # 最初は全てが左ノードにある状態にする。
        left, right = list(range(x.shape[0])), []
        # 説明変数の全ての次元に対して
        for i in range(x.shape[1]):
            feat = x[:, i]  # その次元の値の配列
            for val in feat:
                # 最もよく分割する値を探す
                l, r = self.make_split(feat, val)
                loss = self.make_loss(y[l], y[r], l, r)
                if score > loss:
                    score = loss
                    left = l
                    right = r
                    self.feat_index = i  # 何番目の説明変数で分割するか
                    self.feat_val = val  # 分割する値
        self.score = score  # 最良の分割点のスコア
        return left, right

    def fit(self, x, y):
        '''
        左右の葉を作成する
        '''
        self.left = self.leaf()
        self.right = self.leaf()
        # データを左右の葉に振り分ける
        left, right = self.split_tree(x, y)

        # 左右の葉を学習させる
        if len(left) > 0:
            self.left.fit(x[left], y[left])
        if len(right) > 0:
            self.right.fit(x[right], y[right])

        return self

    def predict(self, x):
        '''
        説明変数から分割した左右のインデックスを取得
        '''
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 左右の葉を実行して結果を作成する
        z = None
        if len(l) > 0 and len(r) > 0:
            left = self.left.predict(x[l])
            right = self.right.predict(x[r])
            z = np.zeros((x.shape[0], left.shape[1]))
            z[l] = left
            z[r] = right
        elif len(l) > 0:
            z = self.left.predict(x)
        elif len(r) > 0:
            z = self.right.predict(x)
        return z


class DescisionStump2:
    def __init__(
            self,
            metric=entropy.gini,
            leaf=zeror.ZeroRule):
        '''
        元のコードから(およそ100倍)高速化したDescitionStump
        '''

        self.metric = metric
        self.leaf = leaf
        self.left = None
        self.right = None
        self.feat_index = 0
        self.feat_val = np.nan
        self.score = np.nan

    def make_split(self, feat, val):
        '''
        featをvalより小さいものとval以上で分割するbooleanインデックスを返す。
        元のコードがfor文を使っているのに対し、booleanインデックスで返すことにより高速化
        '''
        left = feat < val
        right = ~left
        return left, right

    def make_loss(self, y1, y2, l, r):
        '''
        yをy1とy2で分割したときのMetrics関数の重み付き合計を返す。
        '''
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        total = y1.shape[0]+y2.shape[0]
        m1 = self.metric(y1)*(y1.shape[0]/total)
        m2 = self.metric(y2)*(y2.shape[0]/total)
        return m1 + m2

    def split_tree(self, x, y):
        '''
        データを分割して左右の枝に属するインデックスを返す
        元のコードが特徴量の全ての値での分割を試しているののに対し、
        パーセンタイルで20個の候補だけを試すことで高速化
        '''
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        # パーセンタイル、四分位数などを一般化したもの
        # 今回は5%、10%、15%…と区切る
        pctiles = np.linspace(5, 95, 19)
        data_num = x.shape[0]
        # 左右のインデックス
        left = np.ones(x.shape[0], dtype=bool)
        right = ~left
        # 最初は全てが左ノードにある状態にする。
        # もしこの葉にデータが１つしかなければ
        # lossは必ずinfになるのでscore > lossはFalseになる。
        # そうすると、self.valはinfのままとなる。
        # self.rightの予測器は学習していないので
        # 予測時にデータが来るとエラーになってしまうが、
        # 予測時にどんなデータが来てもleftに行くことになるので問題ない

        # 説明変数の全ての次元に対して
        for i in range(x.shape[1]):
            feat = x[:, i]  # その次元の値の配列
            if data_num > 40:
                val_cands = np.percentile(feat, pctiles)
            else:
                val_cands = feat
            for val in val_cands:
                # 最もよく分割する値を探す
                l, r = self.make_split(feat, val)
                loss = self.make_loss(y[l, :], y[r, :], l, r)
                if score > loss:
                    score = loss
                    left = l
                    right = r
                    self.feat_index = i  # 何番目の説明変数で分割するか
                    self.feat_val = val  # 分割する値
        self.score = score  # 最良の分割点のスコア
        return left, right

    def fit(self, x, y):
        '''
        左右の葉を作成する
        '''
        self.left = self.leaf()
        self.right = self.leaf()
        # データを左右の葉に振り分ける
        left, right = self.split_tree(x, y)

        # 左右の葉を学習させる
        if np.any(left):
            self.left.fit(x[left, :], y[left, :])
        if np.any(right):
            self.right.fit(x[right, :], y[right, :])

        return self

    def predict(self, x):
        '''
        説明変数から分割した左右のインデックスを取得
        '''
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        # 左右の葉を実行して結果を作成する
        z = None
        if np.any(l) and np.any(r):
            left = self.left.predict(x[l, :])
            right = self.right.predict(x[r, :])
            z = np.zeros((x.shape[0], left.shape[1]))
            z[l, :] = left
            z[r, :] = right
        elif np.any(l):
            z = self.left.predict(x)
        elif np.any(r):
            z = self.right.predict(x)
        return z


if __name__ == '__main__':

    from pathlib import Path
    from . import evaluate

    gini_model = DescisionStump2()
    gini_result = evaluate.evaluate_classifier(gini_model)

    infogain_model = DescisionStump2(metric=entropy.infgain)
    infogain_result = evaluate.evaluate_classifier(infogain_model)

    reg_model = DescisionStump2(
        metric=entropy.deviation)
    reg_result = evaluate.evaluate_regressor(reg_model)

    result_list = [
        ('gini', gini_result),
        ('infogain', infogain_result),
        ('reg_entropy', reg_result)
    ]
    result = evaluate.merge_result(*result_list)

    print(result)
    p_path = Path(__file__).resolve().parent/'processed_data'/'dstump.csv'
    result.to_csv(p_path, index=None)
