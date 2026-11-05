"""展示航走をどれだけ重く見るか: 展開モデルに「枠＋展示航走だけのモデル」を混ぜる割合 w を変えて、
材料ごとの効き具合（その材料をレース内で平均にならしたとき、3連単の当たりがどれだけ悪くなるか）を測る。

  python tools/tenkai_exhibit.py --work work/full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from research import normalize_place, tri_iter  # noqa: E402
from research_eval import prob_metrics  # noqa: E402
from tenkai import (COURSE, EXHIBIT, clf, load, pair_feats, pair_rows, race_tensor, scenario_probs,  # noqa: E402
                    tri_tenkai, winner_class)

GROUPS = {
    "展示航走": EXHIBIT,
    "選手の力": ["rank", "natWin", "nat2", "nat3", "locWin", "loc2", "loc3", "avgST", "age", "weight", "F", "L", "natWinRank",
              "natWinDiff", "avgSTRank", "natWinVsInner", "wallNatWin", "innerMinNatWin", "innerMeanNatWin", "outerMaxNatWin", "c1NatWinGap"],
    "近況": ["pN90", "pAvgFin", "pWin10", "pTop3", "pCourseWin", "pCourseTop2", "pCourseTop3", "pCourseN", "pVenueWin", "pSetsuAvg",
           "pSetsuN", "pDaysOff", "pSTavg", "pSTstd", "pF180", "stEst", "stEstRank", "stEstVsInner", "stEstVsOuter", "pAvgFinRank",
           "pCourseWinRank", "c1CourseWin"],
    "モーター・ボート": ["motor2", "motor3", "boat2", "boat3", "motor2Rank", "motor2Diff", "mAvgFin", "mWin", "mN"],
}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def neutralize(X, present, feats, cols):
    X = X.copy()
    for c in cols:
        if c not in feats:
            continue
        k = feats.index(c)
        v = np.where(present, X[:, :, k], np.nan)
        m = np.nanmean(v, 1, keepdims=True)
        X[:, :, k] = np.where(present, np.broadcast_to(m, v.shape), X[:, :, k])
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--valid-from", default=None, help="省略時は最新日の60日前")
    ap.add_argument("--valid-to", default="2099-12-31")
    ap.add_argument("--ws", default="0,0.3,0.5,0.7,0.8,0.85,0.9,1.0")
    a = ap.parse_args()
    df, feats = load(Path(a.work))
    if not a.valid_from:
        import pandas as pd
        a.valid_from = (pd.Timestamp(df.date.max()) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    tr = df[df.date < a.valid_from]
    va = df[(df.date >= a.valid_from) & (df.date <= a.valid_to)]
    vc = T.venue_course_rates(tr)
    tr, va = T.fill_vc(tr, vc), T.fill_vc(va, vc)
    Sm = clf().fit(tr[feats].to_numpy(float), winner_class(tr))
    r2 = pair_rows(tr, feats, 2); N2 = clf().fit(r2[pair_feats(feats, 2)].to_numpy(float), r2.y)
    r3 = pair_rows(tr, feats, 3); N3 = clf().fit(r3[pair_feats(feats, 3)].to_numpy(float), r3.y)
    cef = [f for f in COURSE + EXHIBIT if f in feats]
    CE = {k: clf().fit(tr[cef].to_numpy(float), (tr.place <= n).astype(int)) for k, n in (("win", 1), ("top2", 2), ("top3", 3))}
    log("学習おわり")
    ids, present, place, X = race_tensor(va, feats)
    R = len(ids)
    cidx = [feats.index(f) for f in cef]

    def run(Xm):
        S = scenario_probs(Sm, Xm, present)
        tm, _, _ = tri_tenkai(S, N2, N3, Xm, present, feats)
        Xc = Xm[:, :, cidx].reshape(R * 6, -1)
        p = {k: m.predict_proba(Xc)[:, 1].reshape(R, 6) for k, m in CE.items()}
        tc = tri_iter(*normalize_place(p["win"], p["top2"], p["top3"], present), present)
        return tm, tc

    base = run(X)
    neut = {g: run(neutralize(X, present, feats, cols)) for g, cols in GROUPS.items()}
    log("ならし計算おわり")
    res = []
    for w in [float(x) for x in a.ws.split(",")]:
        def comb(tm, tc):
            t = np.exp((1 - w) * np.log(np.maximum(tm, 1e-15)) + w * np.log(np.maximum(tc, 1e-15)))
            return t / t.sum(1, keepdims=True)
        m0 = prob_metrics(comb(*base), place, present)
        imp = {g: round(prob_metrics(comb(*neut[g]), place, present)["tri_logloss"] - m0["tri_logloss"], 4) for g in GROUPS}
        row = {"w": w, **{k: round(v, 4) for k, v in m0.items() if k != "races"}, "importance": imp}
        res.append(row)
        log(json.dumps(row, ensure_ascii=False))
    Path(a.work, "exhibit_weight.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
