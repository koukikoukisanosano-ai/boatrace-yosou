"""予想方法の総当たり比較（ハナハナ予想の research と同じ考え方）

- 各月のレースを「その月より前のデータだけ」で学習したモデルで予想する（ずるの無い成績）
- 開発期間で方法を作り・選び、選ぶのに使っていない本番テスト期間で成績を確かめる
- 確率の方法（的中率・確率の正しさ）と、買い方（券種×点数・期待値の基準・レースの絞り込み）を比べる

  python tools/research.py --work work/full [--dev 2026-04-01:2026-07-31] [--hold 2026-08-01:2026-12-31]
出力: docs/data/research.json（ページの「予想方法の比較」）と work/full/research_preds.pkl
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = list(range(1, 7))
TRI = list(itertools.permutations(range(6), 3))           # 艇の位置(0-5)の並び
TRI_KEY = [f"{a+1}-{b+1}-{c+1}" for a, b, c in TRI]
EXA = list(itertools.permutations(range(6), 2))
QUI = list(itertools.combinations(range(6), 2))
TRIO = list(itertools.combinations(range(6), 3))
TA = np.array([t[0] for t in TRI]); TB = np.array([t[1] for t in TRI]); TC = np.array([t[2] for t in TRI])
ITERS = {"win": 775, "top2": 375, "top3": 500}   # 本番モデルで検証して決めた本数
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- データ
def load(work: Path):
    df = pd.read_csv(work / "features.tsv", sep="\t", low_memory=False)
    meta = ["race", "date", "place", "preview"]
    feats = [c for c in df.columns if c not in meta]
    races = {}
    with open(work / "races.jsonl", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            races[r["id"]] = r
    return df, feats, races


def load_odds(dates: list[str]) -> dict:
    out = {}
    for d in sorted(set(dates)):
        p = ROOT / "data" / "odds" / d[:4] / f"{d.replace('-', '')}.json.gz"
        if not p.exists():
            continue
        day = json.loads(gzip.decompress(p.read_bytes()))
        for k, v in day.items():
            out[f"{d.replace('-', '')}-{k}"] = v
    return out


def race_arrays(df: pd.DataFrame, races: dict):
    """1レース1行の配列（6艇ぶん。欠場の所は使わない）"""
    ids = df.race.unique()
    idx = {r: i for i, r in enumerate(ids)}
    R = len(ids)
    pos = np.full(len(df), -1)
    rr = df.race.map(idx).to_numpy()
    wk = df.waku.to_numpy().astype(int) - 1
    present = np.zeros((R, 6), bool)
    present[rr, wk] = True
    place = np.full((R, 6), 7)
    place[rr, wk] = df.place.to_numpy()
    meta = pd.DataFrame({"race": ids})
    meta["date"] = meta.race.map(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
    return ids, idx, rr, wk, present, place, meta


# ---------------------------------------------------------------- 3連単の確率
def tri_iter(p1, p2, p3, present, iters=12):
    """1着・2着内・3着内の確率 → 3連単（ページの core.js と同じ計算）"""
    eps = 1e-12
    s2t = np.where(present, np.maximum(p2 - p1, 1e-4), 0.0)
    s3t = np.where(present, np.maximum(p3 - p2, 1e-4), 0.0)
    s2t /= s2t.sum(1, keepdims=True)
    s3t /= s3t.sum(1, keepdims=True)
    w2, w3 = s2t.copy(), s3t.copy()
    oh_b = np.eye(6)[TB]; oh_c = np.eye(6)[TC]

    def build():
        sw2 = w2.sum(1, keepdims=True); sw3 = w3.sum(1, keepdims=True)
        return p1[:, TA] * w2[:, TB] / np.maximum(sw2 - w2[:, TA], eps) * w3[:, TC] / np.maximum(sw3 - w3[:, TA] - w3[:, TB], eps)

    tri = build()
    for _ in range(iters):
        m2 = tri @ oh_b; m3 = tri @ oh_c
        w2 *= np.where(present, (s2t / np.maximum(m2, 1e-9)) ** 0.7, 0)
        w3 *= np.where(present, (s3t / np.maximum(m3, 1e-9)) ** 0.7, 0)
        tri = build()
    return tri / tri.sum(1, keepdims=True)


def tri_harville(p1, present, g2=1.0, g3=1.0):
    eps = 1e-12
    w2 = np.where(present, p1 ** g2, 0); w3 = np.where(present, p1 ** g3, 0)
    sw2 = w2.sum(1, keepdims=True); sw3 = w3.sum(1, keepdims=True)
    tri = p1[:, TA] * w2[:, TB] / np.maximum(sw2 - w2[:, TA], eps) * w3[:, TC] / np.maximum(sw3 - w3[:, TA] - w3[:, TB], eps)
    return tri / tri.sum(1, keepdims=True)


def normalize_place(p1, p2, p3, present):
    """レース内で 1着の合計1・2着内の合計2・3着内の合計3 に（core.js と同じ）"""
    p1 = np.where(present, p1, 0); p1 = p1 / p1.sum(1, keepdims=True)

    def scale(p, total):
        q = np.where(present, p, 0.0)
        for _ in range(20):
            fixed = q >= 0.999
            sf = fixed.sum(1, keepdims=True)
            free = np.where(fixed, 0, q).sum(1, keepdims=True)
            k = (total - sf) / np.maximum(free, 1e-12)
            q = np.where(fixed, 0.999, np.minimum(0.999, q * k))
            q = np.where(present, q, 0)
        return q
    n = present.sum(1, keepdims=True)
    t2 = scale(p2, np.minimum(2, n)); t3 = scale(p3, np.minimum(3, n))
    t2 = np.maximum(t2, p1); t3 = np.maximum(t3, t2)
    return p1, t2, t3


# ---------------------------------------------------------------- 確率の方法
def fit_gbm(tr, feats, targets=("win", "top2", "top3"), weight=None, iters=None):
    vc = T.venue_course_rates(tr)
    trf = T.fill_vc(tr, vc)
    X = trf[feats].to_numpy(float)
    models = {}
    for name in targets:
        k = {"win": 1, "top2": 2, "top3": 3}[name]
        clf = T.make_clf((iters or ITERS)[name])
        clf.fit(X, (trf.place <= k).astype(int).to_numpy(), sample_weight=weight)
        models[name] = clf
    return vc, models


def predict_gbm(te, feats, vc, models):
    tef = T.fill_vc(te, vc)
    X = tef[feats].to_numpy(float)
    return {k: m.predict_proba(X)[:, 1] for k, m in models.items()}


def to_race(values, rr, wk, R):
    a = np.zeros((R, 6))
    a[rr, wk] = values
    return a


class CLogit:
    """条件付きロジット（レースの中で6艇を比べる線形モデル）。上位3着の順番で学習（Plackett-Luce）"""

    def __init__(self, feats, l2=1.0):
        self.feats = feats
        self.l2 = l2

    def _x(self, df):
        X = df[self.feats].to_numpy(float)
        Z = (X - self.mu) / self.sd
        miss = np.isnan(Z)
        Z[miss] = 0.0
        return np.concatenate([Z, miss[:, self.miss_cols].astype(float)], 1)

    def fit(self, tr, vc):
        trf = T.fill_vc(tr, vc)
        X = trf[self.feats].to_numpy(float)
        self.mu = np.nanmean(X, 0); self.sd = np.nanstd(X, 0) + 1e-9
        self.mu = np.nan_to_num(self.mu); self.sd[~np.isfinite(self.sd)] = 1.0
        miss_rate = np.isnan(X).mean(0)
        self.miss_cols = np.where((miss_rate > 0.01) & (miss_rate < 0.99))[0]
        Z = self._x(trf)
        ids, idx, rr, wk, present, place, _ = race_arrays(trf, None)
        R, F = len(ids), Z.shape[1]
        Xr = np.zeros((R, 6, F)); Xr[rr, wk] = Z
        order = np.argsort(np.where(present, place, 99), 1)[:, :3]   # 1〜3着の艇
        ok = (np.take_along_axis(place, order, 1) == np.array([1, 2, 3])).all(1)
        Xr, present, order = Xr[ok], present[ok], order[ok]
        n = len(Xr)

        def f(w):
            s = Xr @ w
            s = np.where(present, s, -1e9)
            loss = 0.0; g = np.zeros_like(w)
            avail = present.copy()
            for k in range(3):
                sk = np.where(avail, s, -1e9)
                lse = logsumexp(sk, 1)
                o = order[:, k]
                loss -= (s[np.arange(n), o] - lse).sum()
                pr = np.exp(sk - lse[:, None])
                g -= (Xr[np.arange(n), o] - np.einsum("rb,rbf->rf", pr, Xr)).sum(0)
                avail[np.arange(n), o] = False
            loss = loss / n + self.l2 * (w @ w) / 2 / 100
            g = g / n + self.l2 * w / 100
            return loss, g

        res = minimize(f, np.zeros(F), jac=True, method="L-BFGS-B", options={"maxiter": 300})
        self.w = res.x
        return self

    def strength(self, te, vc):
        tef = T.fill_vc(te, vc)
        return self._x(tef) @ self.w


def pl_from_strength(s, present):
    """強さ s → 1着の確率と3連単（Plackett-Luce）"""
    e = np.where(present, np.exp(s - np.where(present, s, -np.inf).max(1, keepdims=True)), 0)
    p1 = e / e.sum(1, keepdims=True)
    return p1, tri_harville(p1, present)


# ---------------------------------------------------------------- 採点
def market_probs(odds_list, present, n):
    """オッズ → 市場の確率（逆数を正規化）"""
    a = np.array([[np.nan if v is None else v for v in o] if o else [np.nan] * n for o in odds_list], float)
    inv = np.where(np.isfinite(a) & (a > 0), 1 / a, 0)
    s = inv.sum(1, keepdims=True)
    return np.where(s > 0, inv / np.maximum(s, 1e-12), np.nan), a


def eval_probs(name, p1, tri, place, present, payouts):
    R = len(p1)
    winner = np.argmax(place == 1, 1)
    has_w = (place == 1).any(1)
    order = np.argsort(np.where(present, place, 99), 1)[:, :3]
    true_tri = np.array([TRI.index(tuple(o)) if (np.take_along_axis(place[i:i+1], o[None], 1) == [[1, 2, 3]]).all() else -1 for i, o in enumerate(order)])
    ok = has_w & (true_tri >= 0)
    pw = p1[np.arange(R), winner]
    pt = tri[np.arange(R), np.maximum(true_tri, 0)]
    top1 = np.argmax(p1, 1) == winner
    rank_tri = np.argsort(-tri, 1)
    hit_t1 = rank_tri[:, 0] == true_tri
    hit_t5 = (rank_tri[:, :5] == true_tri[:, None]).any(1)
    # 2連単・3連複・2連複（3連単から合算）
    exa = np.zeros((R, 30)); trio = np.zeros((R, 20)); qui = np.zeros((R, 15))
    for k, (a, b, c) in enumerate(TRI):
        exa[:, EXA.index((a, b))] += tri[:, k]
        trio[:, TRIO.index(tuple(sorted((a, b, c))))] += tri[:, k]
        qui[:, QUI.index(tuple(sorted((a, b))))] += tri[:, k]
    tw = np.where(ok, true_tri, 0)
    t_exa = np.array([EXA.index((TRI[i][0], TRI[i][1])) for i in tw])
    t_trio = np.array([TRIO.index(tuple(sorted(TRI[i]))) for i in tw])
    t_qui = np.array([QUI.index(tuple(sorted(TRI[i][:2]))) for i in tw])
    m = ok
    return {
        "method": name, "races": int(m.sum()),
        "win_top1": float(top1[m].mean()),
        "win_logloss": float(-np.log(np.clip(pw[m], 1e-9, 1)).mean()),
        "tri_logloss": float(-np.log(np.clip(pt[m], 1e-12, 1)).mean()),
        "tri_top1": float(hit_t1[m].mean()), "tri_top5": float(hit_t5[m].mean()),
        "exa_top1": float((np.argmax(exa, 1) == t_exa)[m].mean()),
        "trio_top1": float((np.argmax(trio, 1) == t_trio)[m].mean()),
        "qui_top1": float((np.argmax(qui, 1) == t_qui)[m].mean()),
    }, exa, trio, qui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work" / "full"))
    ap.add_argument("--first-test", default="", help="予想し直す最初の月（空なら最新月の7か月前）")
    ap.add_argument("--dev-end", default="", help="開発期間の最後の月（空なら最新月の4か月前）")
    ap.add_argument("--quick", action="store_true", help="木の本数を減らして速く（試し用）")
    a = ap.parse_args()
    work = Path(a.work)
    df, feats, races = load(work)
    df = df.sort_values(["date", "race", "waku"]).reset_index(drop=True)
    all_months = sorted({d[:7] for d in df.date})
    if not a.first_test:   # 開発4か月＋本番テスト4か月（最新月は途中まで）
        a.first_test = all_months[max(0, len(all_months) - 8)]
    if not a.dev_end:
        a.dev_end = all_months[max(0, len(all_months) - 5)]
    months = sorted({d[:7] for d in df.date if d[:7] >= a.first_test})
    log(f"データ {df.race.nunique()} レース / 検証する月 {months}")
    iters = {k: max(60, v // 4) for k, v in ITERS.items()} if a.quick else ITERS

    preds = []   # 月ごとの予想（レース単位）
    for m in months:
        tr = df[df.date < f"{m}-01"]
        te = df[df.date.str.startswith(m)].reset_index(drop=True)
        ids, idx, rr, wk, present, place, meta = race_arrays(te, races)
        R = len(ids)
        out = {"month": m, "ids": ids, "present": present, "place": place}
        # 1) 勾配ブースティング（今の方法）
        vc, models = fit_gbm(tr, feats, iters=iters)
        pg = predict_gbm(te, feats, vc, models)
        out["gbm"] = {k: to_race(v, rr, wk, R) for k, v in pg.items()}
        # 2) 新しい日ほど重く（半減期90日）
        age = (pd.Timestamp(f"{m}-01") - pd.to_datetime(tr.date)).dt.days.to_numpy()
        wgt = 0.5 ** (age / 90)
        vc2, models2 = fit_gbm(tr, feats, weight=wgt, iters=iters)
        pg2 = predict_gbm(te, feats, vc2, models2)
        out["gbm_rw"] = {k: to_race(v, rr, wk, R) for k, v in pg2.items()}
        # 3) 条件付きロジット
        cl = CLogit(feats).fit(tr, vc)
        out["clogit_s"] = to_race(cl.strength(te, vc), rr, wk, R)
        preds.append(out)
        log(f"{m}: 学習 {tr.race.nunique()} レース → 予想 {R} レース")

    with open(work / "research_preds.pkl", "wb") as f:
        pickle.dump({"preds": preds, "months": months, "dev_end": a.dev_end}, f)
    log("予想を保存しました:", work / "research_preds.pkl")


if __name__ == "__main__":
    main()
