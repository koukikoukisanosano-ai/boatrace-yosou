"""6艇を比べて学ぶモデル（レースごとのソフトマックス＝条件付きロジットを勾配ブースティングで）

  1着: 各艇の強さ s1 → P(a が1着) = exp(s1_a) / Σ exp(s1)
  2着: 1着 a が決まったときの残りの艇の強さ s2(b|a) → P(b が2着 | a) = softmax_{b≠a}(s2)
  3着: 残りの艇で s1 のソフトマックス（Plackett-Luce）

学習は「ニュートン法のブースティング」: 毎回、今の確率との差（勾配）を HistGradientBoosting の木1本で近似して足していく。
木は sklearn の形なので、ページ（core.js）でも同じ書き出し・評価が使える。
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _group_softmax(s, gid, n_groups):
    m = np.full(n_groups, -np.inf)
    np.maximum.at(m, gid, s)
    e = np.exp(s - m[gid])
    z = np.zeros(n_groups)
    np.add.at(z, gid, e)
    return e / z[gid]


class SoftmaxBoost:
    def __init__(self, rounds=400, lr=0.1, leaves=31, min_leaf=40, l2=1.0, log=None):
        self.rounds, self.lr, self.leaves, self.min_leaf, self.l2 = rounds, lr, leaves, min_leaf, l2
        self.trees = []
        self.log = log

    def fit(self, X, gid, y):
        """X: 行×特徴量 / gid: グループ（レース）番号 0..G-1 / y: その行が正解（1）か"""
        G = int(gid.max()) + 1
        s = np.zeros(len(X))
        t0 = time.time()
        for it in range(self.rounds):
            p = _group_softmax(s, gid, G)
            h = np.maximum(p * (1 - p), 1e-6)
            target = (y - p) / h
            reg = HistGradientBoostingRegressor(max_iter=1, learning_rate=1.0, max_leaf_nodes=self.leaves, min_samples_leaf=self.min_leaf,
                                                l2_regularization=self.l2, early_stopping=False, random_state=it)
            reg.fit(X, np.clip(target, -20, 20), sample_weight=h)
            self.trees.append(reg)
            s += self.lr * reg.predict(X)
            if self.log and (it + 1) % 100 == 0:
                ll = -np.log(np.maximum(p[y == 1], 1e-12)).mean()
                self.log(f"    {it + 1} 本 logloss {ll:.4f}（{time.time() - t0:.0f}秒）")
        return self

    def score(self, X):
        s = np.zeros(len(X))
        for reg in self.trees:
            s += self.lr * reg.predict(X)
        return s


def race_tri(s1, s2, present):
    """s1: (R,6) 1着の強さ / s2: (R,6,6) [a, b] 1着aのときのbの2着の強さ → 3連単 (R,120)"""
    R = s1.shape[0]
    neg = -1e9
    a1 = np.where(present, s1, neg)
    p1 = np.exp(a1 - a1.max(1, keepdims=True)); p1 /= p1.sum(1, keepdims=True)
    tri = np.zeros((R, 120))
    perms = list(itertools.permutations(range(6), 3))
    p2 = np.zeros((R, 6, 6))
    for a in range(6):
        m = present.copy(); m[:, a] = False
        v = np.where(m, s2[:, a, :], neg)
        e = np.exp(v - v.max(1, keepdims=True)) * m
        p2[:, a, :] = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    for k, (a, b, c) in enumerate(perms):
        m = present.copy(); m[:, a] = False; m[:, b] = False
        v = np.where(m, s1, neg)
        e = np.exp(v - v.max(1, keepdims=True)) * m
        p3 = e[:, c] / np.maximum(e.sum(1), 1e-12)
        tri[:, k] = p1[:, a] * p2[:, a, b] * p3
    tri *= present[:, [p[0] for p in perms]] & present[:, [p[1] for p in perms]] & present[:, [p[2] for p in perms]]
    return tri / np.maximum(tri.sum(1, keepdims=True), 1e-12)


class LgbSoftmax:
    """同じことを LightGBM（自作の目的関数）で速く。GitHub の計算機用（pip install lightgbm）"""

    def __init__(self, rounds=600, lr=0.05, leaves=31, min_leaf=40, l2=1.0, log=None):
        self.p = dict(learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf, lambda_l2=l2, verbose=-1, num_threads=0,
                      feature_fraction=0.9, bagging_fraction=1.0, max_bin=255)
        self.rounds, self.log = rounds, log

    def fit(self, X, gid, y):
        import lightgbm as lgb
        G = int(gid.max()) + 1

        def obj(preds, data):
            p = _group_softmax(preds, gid, G)
            return p - y, np.maximum(p * (1 - p), 1e-6)

        ds = lgb.Dataset(X, label=y, free_raw_data=False)
        t0 = time.time()
        self.b = lgb.train({**self.p, "objective": obj}, ds, num_boost_round=self.rounds)
        if self.log:
            s = self.b.predict(X, raw_score=True)
            p = _group_softmax(s, gid, G)
            self.log(f"    学習 logloss {-np.log(np.maximum(p[y == 1], 1e-12)).mean():.4f}（{time.time() - t0:.0f}秒）")
        return self

    def score(self, X):
        return self.b.predict(X, raw_score=True)
