import numpy as np
from . import support

class Linear:
    def __init__(self,epochs=20,lr=0.01,earlystop=None):
        self.epochs=epochs
        self.lr=lr
        self.earlystop=earlystop
        self.beta=None
        self.norm=None

    def fitnorm(self,x,y):
        # 学習の前にデータに含まれる値の範囲を0から1に正規化しておく
        # そのためのパラメータを保存しておく
        self.norm = np.zeros( (x.shape[ 1 ] + 1,2))
        self.norm[ 0,0 ]=np.min( y ) #目的変数の最小値
        self.norm[ 0,1 ]=np.max( y ) # 目的変数の最大値
        self.norm[ 1:,0]=np.min(x,axis=0) # 説明変数の最小値
        self.norm[ 1:,1]=np.max(x,axis=0) # 説明変数の最大値

    def normalize(self,x,y=None):
        # データに含まれる値の範囲を0から1に正規化する
        l=self.norm[1:,1]-self.norm[1:,0] # 最大値から最小値を引く
        # 最大値と最小値が等しい場合(その説明変数がすべて定数の場合)
        # lには1を入れておく。
        l[l==0] = 1
        # 説明変数-最小値をlで割る。
        p=(x-self.norm[1:,0])/l 
        q=y
        # もしyがNoneでなく、yの最小値と最大値が等しい場合
        if y is not None and not self.norm[0,1]==self.norm[0,0]:
            q=(y-self.norm[0,0])/(self.norm[0,1]-self.norm[0,0])
        return p,q

    def r2(self,y,z):
        # EarlyStopping用にR2スコアを計算する
        y = y.reshape( (-1,))
        z = z.reshape( (-1,))
        mn=( (y-z)**2 ).sum(axis=0)
        dn=((y-y.mean())**2).sum(axis=0)
        if dn==0:
            return np.inf
        return 1.0-mn/dn

    def fit(self,x,y):
        # 勾配降下法による線形回帰係数の推定を行う
        # 最初に、データに含まれる値の範囲を0から1に正規化する。
        self.fitnorm( x,y)
        x,y = self.normalize(x,y)

        # 線形回帰・・・配列の最初の値がy=ax+bのbに、続く値がaになる
        self.beta=np.zeros( (x.shape[1] + 1, ))
        # 繰り返し学習を行う
        for _ in range(self.epochs):
            # 1エポック内でデータを繰り返す
            for p,q in zip(x,y):
                # 現在のモデルによる出力から勾配を求める
                z = self.predict( p.reshape( (1,-1) ),normalized=True)
                z = z.reshape( (1,))
                err=(z-q)*self.lr
                delta = p*err
                # モデルを更新する
                self.beta[ 0 ] -= err
                self.beta[ 1: ] -= delta
            # EarlyStoppingが有効なら
            if self.earlystop is not None:
                # スコアを求めて一定以上なら終了
                z = self.predict(x,normalized=True)
                s=self.r2(y,z)
                if self.earlystop <= s:
                    break

        return self

    def predict(self,x,normalized=False):
        # 線形回帰モデルを実行する
        # まずは値の範囲を0から1に正規化する
        if not normalized:
            x, _=self.normalize(x)
        # 結果を計算する
        z= np.zeros( (x.shape[0],1) )+self.beta[0]
        for i in range( x.shape[1] ):
            c=x[:,i]*self.beta[i+1]
            z += c.reshape( (-1,1) )
        # 0から1に正規化した値を返す
        if not normalized:
            z = z*(self.norm[0,1] - self.norm[0,0]) + self.norm[0,0]
        return z

    def __str__(self):
        # モデルの内容を文字列にする
        if type(self.beta) is not type (None):
            s = [ '%f'%self.beta[0] ]
            e = [ ' + feat[%d] * %f'%( i+1,j )for i,j in enumerate( self.beta[1:])]
            s.extend(e)
            return ''.join(s)
        else:
            return '0.0'

class LinearNE:
    def __init__(self,epochs=20,lr=0.01,earlystop=None):
        '''
        正規方程式によって回帰係数を求める
        正規方程式について:
        https://manabitimes.jp/math/1128
        https://www.slideshare.net/wosugi/ss-79624897
        '''
        self.epochs=epochs
        self.lr=lr
        self.earlystop=earlystop
        self.beta=None
        self.norm=None

    def fitnorm(self,x,y):
        # 学習の前にデータに含まれる値の範囲を0から1に正規化しておく
        # そのためのパラメータを保存しておく
        self.norm = np.zeros( (x.shape[ 1 ] + 1,2))
        self.norm[ 0,0 ]=np.min( y ) #目的変数の最小値
        self.norm[ 0,1 ]=np.max( y ) # 目的変数の最大値
        self.norm[ 1:,0]=np.min(x,axis=0) # 説明変数の最小値
        self.norm[ 1:,1]=np.max(x,axis=0) # 説明変数の最大値

    def normalize(self,x,y=None):
        # データに含まれる値の範囲を0から1に正規化する
        l=self.norm[1:,1]-self.norm[1:,0] # 最大値から最小値を引く
        # 最大値と最小値が等しい場合(その説明変数がすべて定数の場合)
        # lには1を入れておく。
        l[l==0] = 1
        # 説明変数-最小値をlで割る。
        p=(x-self.norm[1:,0])/l 
        q=y
        # もしyがNoneでなく、yの最小値と最大値が等しい場合
        if y is not None and not self.norm[0,1]==self.norm[0,0]:
            q=(y-self.norm[0,0])/(self.norm[0,1]-self.norm[0,0])
        return p,q

    def r2(self,y,z):
        # EarlyStopping用にR2スコアを計算する
        y = y.reshape( (-1,))
        z = z.reshape( (-1,))
        mn=( (y-z)**2 ).sum(axis=0)
        dn=((y-y.mean())**2).sum(axis=0)
        if dn==0:
            return np.inf
        return 1.0-mn/dn

    def fit(self,x,y):
        # 正規方程式によって線形回帰係数の推定を行う
        # 最初に、データに含まれる値の範囲を0から1に正規化する。
        self.fitnorm(x,y)
        x,y = self.normalize(x,y)

        # データ行列を作成する。
        # xx:縦データ数×横説明変数種類+1
        xx=np.ones((x.shape[0],x.shape[1]+1))
        xx[:,1:]=x

        # 正規方程式を解くことでbetaを求める
        self.beta=( np.linalg.pinv(xx.T @ xx) @ xx.T @ y )[:,0]

        return self

    def fit2(self,x,y):
        # 勾配降下法による線形回帰係数の推定を行う
        # 最初に、データに含まれる値の範囲を0から1に正規化する。
        self.fitnorm(x,y)
        x,y = self.normalize(x,y)

        # ヤコビアンを計算する
        # J:縦データ数×横説明変数種類+1
        J=np.ones((x.shape[0],x.shape[1]+1))
        J[:,1:]=x

        # 線形回帰・・・配列の最初の値がy=ax+bのbに、続く値がaになる
        self.beta=np.zeros( (x.shape[1] + 1, ))
        # 繰り返し学習を行う
        for _ in range(self.epochs*100):
            # まず残差を計算する
            r=self.predict2(x,normalized=True)-y
            # 勾配はJ.T @ r
            delta = (J.T @ r ) * self.lr*1e-3
            self.beta -= delta[:,0]
            # EarlyStoppingが有効なら
            if self.earlystop is not None:
                # スコアを求めて一定以上なら終了
                z = self.predict(x,normalized=True)
                s=self.r2(y,z)
                if self.earlystop <= s:
                    break

        return self

    def predict(self,x,normalized=False):
        # 線形回帰モデルを実行する
        # まずは値の範囲を0から1に正規化する
        if not normalized:
            x, _=self.normalize(x)
        # 結果を計算する
        z= np.zeros( (x.shape[0],1) )+self.beta[0]
        z += (x @ self.beta[1:]).reshape((-1,1))
        # 0から1に正規化した値を返す
        if not normalized:
            z = z*(self.norm[0,1] - self.norm[0,0]) + self.norm[0,0]
        return z

    def __str__(self):
        # モデルの内容を文字列にする
        if type(self.beta) is not type (None):
            s = [ '%f'%self.beta[0] ]
            e = [ ' + feat[%d] * %f'%( i+1,j )for i,j in enumerate( self.beta[1:])]
            s.extend(e)
            return ''.join(s)
        else:
            return '0.0'



if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path

    from . import evaluate

    ln=LinearNE()
    ln_result=evaluate.evaluate_regressor(ln)

    result_list=[
        ('LinearNE',ln_result)
    ]
    result=evaluate.merge_result(*result_list)

    print(result)
    p_path=Path(__file__).resolve().parent/'processed_data'/'linear.csv'
    result.to_csv(p_path,index=None)