"""公平な検証（その1）: ある月を、その月より前のデータだけで学習したモデルで予想する

  python tools/fair_fit.py --work work/full --month 2026-05 --out fair_2026-05.pkl
モデル: 展開モデル（標準・大きめ）・枠＋展示・機械学習。オッズは使わない
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from research import fit_gbm, normalize_place, tri_iter  # noqa: E402
from research_exacta import load, log  # noqa: E402
from tenkai import COURSE, EXHIBIT, clf, pair_feats, pair_rows, race_tensor, scenario_probs, tri_tenkai, winner_class  # noqa: E402

# 名前: (木の本数, 学習率, 葉の数, 新しい日を重く見る半減期[日] or None)
VARIANTS = {"tenkai": (300, 0.08, 15, None), "tenkai_big": (600, 0.05, 31, None)}
# 試して採用しなかったもの（--variants で指定すれば作れる）
EXTRA = {"tenkai_xl": (1000, 0.03, 31, None), "tenkai_rw": (600, 0.05, 31, 180),
         "tenkai_l63": (600, 0.05, 63, None), "tenkai_slow": (1500, 0.02, 31, None)}


def fit_rank(trf, X, present, feats, rounds=400, lr=0.1, leaves=31):
    """6艇を比べて学ぶモデル（rank_model.py）で 3連単の確率を出す"""
    from rank_model import SoftmaxBoost, race_tri
    d = trf[trf.race.isin(trf[trf.place == 1].groupby("race").size().pipe(lambda x: x[x == 1]).index)]
    gid = d.race.astype("category").cat.codes.to_numpy()
    m1 = SoftmaxBoost(rounds, lr, leaves, log=log).fit(d[feats].to_numpy(float), gid, (d.place == 1).to_numpy(float))
    log("  順位づけ 1着: 学習")
    r2 = pair_rows(d, feats, 2)
    pf = [f for f in pair_feats(feats, 2) if f != "tech_w"]
    g2 = r2.race.astype("category").cat.codes.to_numpy()
    m2 = SoftmaxBoost(rounds, lr, leaves, log=log).fit(r2[pf].to_numpy(float), g2, r2.y.to_numpy(float))
    log("  順位づけ 2着: 学習")
    R = X.shape[0]
    s1 = m1.score(X.reshape(R * 6, -1)).reshape(R, 6)
    from tenkai import PEER
    fi = {f: k for k, f in enumerate(feats)}
    ci = fi["course"]
    s2 = np.zeros((R, 6, 6))
    for a in range(6):
        wa = X[:, a, [fi[k] for k in PEER]]
        rows = [np.concatenate([X[:, b, :], wa, (X[:, b, ci] - X[:, a, ci])[:, None]], 1) for b in range(6)]
        s2[:, a, :] = m2.score(np.concatenate(rows)).reshape(6, R).T
    return race_tri(s1, s2, present)


def recency(dates, end, half):
    import pandas as pd
    age = (pd.Timestamp(end) - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--train-from", default="2024-11-01")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="", help="展開モデルの種類（空なら全部）")
    a = ap.parse_args()
    df, feats = load(Path(a.work))
    df = df[df.date >= a.train_from]
    tr = df[df.date < f"{a.month}-01"]
    te = df[df.date.str.startswith(a.month)]
    log(f"{a.month}: 学習 {tr.race.nunique()} レース（〜{tr.date.max()}）→ 予想 {te.race.nunique()} レース")
    vc = T.venue_course_rates(tr)
    trf, tef = T.fill_vc(tr, vc), T.fill_vc(te, vc)
    ids, present, place, X = race_tensor(tef, feats)
    R = len(ids)
    out = {"month": a.month, "ids": ids, "present": present, "place": place}
    import pandas as pd
    for name, (n, lr, leaves, half) in {**VARIANTS, **EXTRA}.items():
        if name in EXTRA and name not in a.variants.split(","):   # 本番と同じもの（VARIANTS）はいつも作る。試しのものは指定したときだけ
            continue
        mk = lambda: clf(n, lr, leaves)  # noqa: E731
        w = (lambda d: recency(d, f"{a.month}-01", half)) if half else (lambda d: None)
        Sm = mk().fit(trf[feats].to_numpy(float), winner_class(trf), sample_weight=w(trf.date))
        r2 = pair_rows(trf, feats, 2)
        N2 = mk().fit(r2[pair_feats(feats, 2)].to_numpy(float), r2.y, sample_weight=w(pd.to_datetime(r2.race.str[:8]).dt.strftime("%Y-%m-%d"))); del r2
        r3 = pair_rows(trf, feats, 3)
        N3 = mk().fit(r3[pair_feats(feats, 3)].to_numpy(float), r3.y, sample_weight=w(pd.to_datetime(r3.race.str[:8]).dt.strftime("%Y-%m-%d"))); del r3
        out[name], _, _ = tri_tenkai(scenario_probs(Sm, X, present), N2, N3, X, present, feats)
        log(f"  展開モデル {name}")
    if a.variants and "rank" in a.variants.split(","):
        out["rank"] = fit_rank(trf, X, present, feats)
        log("  順位づけモデル")
    cef = [f for f in COURSE + EXHIBIT if f in feats]
    Xc = X[:, :, [feats.index(f) for f in cef]].reshape(R * 6, -1)
    p = {k: clf().fit(trf[cef].to_numpy(float), (trf.place <= m).astype(int)).predict_proba(Xc)[:, 1].reshape(R, 6)
         for k, m in (("win", 1), ("top2", 2), ("top3", 3))}
    out["ce"] = tri_iter(*normalize_place(p["win"], p["top2"], p["top3"], present), present)
    log("  枠＋展示")
    _, models = fit_gbm(tr, feats)
    Xf = X.reshape(R * 6, -1)
    pg = {k: m.predict_proba(Xf)[:, 1].reshape(R, 6) for k, m in models.items()}
    out["gbm"] = tri_iter(*normalize_place(pg["win"], pg["top2"], pg["top3"], present), present)
    log("  機械学習")
    for k in list(out):
        if k not in ("month", "ids", "present", "place"):
            out[k] = out[k].astype(np.float32)
    pickle.dump(out, open(a.out, "wb"))
    log("保存", a.out)


if __name__ == "__main__":
    main()
