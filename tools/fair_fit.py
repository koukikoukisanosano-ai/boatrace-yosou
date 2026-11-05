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

VARIANTS = {"tenkai": (300, 0.08, 15), "tenkai_big": (600, 0.05, 31)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--train-from", default="2024-11-01")
    ap.add_argument("--out", required=True)
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
    for name, (n, lr, leaves) in VARIANTS.items():
        mk = lambda: clf(n, lr, leaves)  # noqa: E731
        Sm = mk().fit(trf[feats].to_numpy(float), winner_class(trf))
        r2 = pair_rows(trf, feats, 2); N2 = mk().fit(r2[pair_feats(feats, 2)].to_numpy(float), r2.y); del r2
        r3 = pair_rows(trf, feats, 3); N3 = mk().fit(r3[pair_feats(feats, 3)].to_numpy(float), r3.y); del r3
        out[name], _, _ = tri_tenkai(scenario_probs(Sm, X, present), N2, N3, X, present, feats)
        log(f"  展開モデル {name}")
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
    for k in ("tenkai", "tenkai_big", "ce", "gbm"):
        out[k] = out[k].astype(np.float32)
    pickle.dump(out, open(a.out, "wb"))
    log("保存", a.out)


if __name__ == "__main__":
    main()
