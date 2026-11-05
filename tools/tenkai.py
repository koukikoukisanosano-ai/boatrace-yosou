"""展開モデル: 「誰が・どの決まり手で勝つか」→「その展開で誰が2着か」→「3着は誰か」で3連単を作る

  S  : 艇ごとに [負け, 逃げ, 差し, まくり, まくり差し, 抜き・恵まれ] の確率（多クラス）
  N2 : 1着の艇と決まり手が決まったとき、残りの艇が2着になる確率（1着艇との並び・選手の組み合わせを使う）
  N3 : 1・2着が決まったとき、残りの艇が3着になる確率
  3連単 P(a,b,c) = Σ_決まり手 S(a,t) × N2(b | a,t) × N3(c | a,b)

  python tools/tenkai.py --work work/full [--valid-from 2026-09-04]   （検証）
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TECH = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}   # 恵まれは抜きと同じ組
TECH_NAMES = ["", "逃げ", "差し", "まくり", "まくり差し", "抜き・恵まれ"]
# 1着・2着の艇から付け足す特徴（相手として見る）
PEER = ["course", "rank", "natWin", "exTime", "exST", "exTimeRank", "exSTRank", "motor2", "avgST", "stEst", "pCourseWin", "pAvgFin", "age"]
EXHIBIT = ["exTime", "exST", "exSTF", "tilt", "weightAdj", "propNew", "partsRing", "partsMajor", "exTimeRank", "exTimeDiff",
           "exTimeToBest", "exSTRank", "exSTDiff", "exSTvsInner", "exSTvsOuter", "exSTvsC1", "outerThreat", "exTimeVsInner", "c1ExST"]
COURSE = ["jcd", "waku", "course", "courseChanged", "vcWin", "vcTop2", "vcTop3", "nBoats", "rno", "grade", "raceType", "fixed",
          "wind", "windSin", "windCos", "wave"]
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def clf(n=300, lr=0.08, leaves=15):
    return HistGradientBoostingClassifier(learning_rate=lr, max_iter=n, max_leaf_nodes=leaves, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def load(work: Path):
    df = pd.read_csv(work / "features.tsv", sep="\t", low_memory=False)
    meta = ["race", "date", "place", "preview"]
    feats = [c for c in df.columns if c not in meta]
    tech = {}
    for ln in open(work / "races.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        tech[r["id"]] = TECH.get(r.get("tech") or 0, 0)
    df["tech"] = df.race.map(tech).fillna(0).astype(int)
    return df, feats


# ---------------------------------------------------------------- 学習用の表
def winner_class(df):
    return np.where(df.place == 1, df.tech, 0)


def pair_rows(df, feats, stage):
    """stage=2: 1着が決まったときの2着候補 / stage=3: 1・2着が決まったときの3着候補（正解の順位だけで作る）"""
    d = df[["race", "place", "tech"] + [f for f in feats if f not in ("race", "place", "tech")]]
    # 同着のレースは除く
    n1 = d[d.place == 1].groupby("race").size(); n2 = d[d.place == 2].groupby("race").size()
    ok = set(n1[n1 == 1].index) & set(n2[n2 == 1].index)
    d = d[d.race.isin(ok)]
    first = d[d.place == 1].set_index("race")
    second = d[d.place == 2].set_index("race")
    cand = d[d.place > (1 if stage == 2 else 2)].copy()
    cand = cand[cand.race.isin(first.index) & (cand.race.isin(second.index) if stage == 3 else True)]
    out = cand.copy()
    for k in PEER:
        out[f"w_{k}"] = first.loc[cand.race, k].to_numpy()
    out["dcw"] = out["course"] - out["w_course"]
    out["tech_w"] = first.loc[cand.race, "tech"].to_numpy() if "tech" in first else 0
    if stage == 3:
        for k in PEER:
            out[f"s_{k}"] = second.loc[cand.race, k].to_numpy()
        out["dcs"] = out["course"] - out["s_course"]
        out["y"] = (out.place == 3).astype(int)
    else:
        out["y"] = (out.place == 2).astype(int)
    return out


def pair_feats(feats, stage):
    f = list(feats) + [f"w_{k}" for k in PEER] + ["dcw"]
    if stage == 2:
        f += ["tech_w"]
    else:
        f += [f"s_{k}" for k in PEER] + ["dcs"]
    return f


# ---------------------------------------------------------------- 予想（Python 版。ページは core.js で同じ計算）
def race_tensor(df, feats):
    ids = df.race.unique()
    idx = {r: i for i, r in enumerate(ids)}
    rr = df.race.map(idx).to_numpy(); wk = df.waku.to_numpy().astype(int) - 1
    R = len(ids)
    present = np.zeros((R, 6), bool); present[rr, wk] = True
    place = np.full((R, 6), 7); place[rr, wk] = df.place.to_numpy()
    X = np.full((R, 6, len(feats)), np.nan); X[rr, wk] = df[feats].to_numpy(float)
    return ids, present, place, X


def peer_block(X, feats, i):
    fi = {f: k for k, f in enumerate(feats)}
    return X[:, i, [fi[k] for k in PEER]]


def tri_tenkai(S, N2, N3, X, present, feats):
    """S: (R,6,6) 艇×クラスの確率（レースで正規化済み）。3連単 (R,120) を返す"""
    R = X.shape[0]
    fi = {f: k for k, f in enumerate(feats)}
    ci = fi["course"]
    perms = list(itertools.permutations(range(6), 3))
    tri = np.zeros((R, 120))
    # 2着: 1着 a・決まり手 t ごとに
    p2 = np.zeros((R, 6, 6, 6))   # [r, a, t, b]
    for a in range(6):
        wa = peer_block(X, feats, a)
        for t in range(1, 6):
            rows, keys = [], []
            for b in range(6):
                if b == a:
                    continue
                f = np.concatenate([X[:, b, :], wa, (X[:, b, ci] - X[:, a, ci])[:, None], np.full((R, 1), t)], 1)
                rows.append(f); keys.append(b)
            P = N2.predict_proba(np.concatenate(rows))[:, 1].reshape(len(keys), R).T
            P = np.where(present[:, keys], P, 0)
            P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
            p2[:, a, t, keys] = P
    # 3着: 1着 a・2着 b ごとに
    p3 = np.zeros((R, 6, 6, 6))
    for a in range(6):
        wa = peer_block(X, feats, a)
        for b in range(6):
            if b == a:
                continue
            sb = peer_block(X, feats, b)
            rows, keys = [], []
            for c in range(6):
                if c in (a, b):
                    continue
                f = np.concatenate([X[:, c, :], wa, (X[:, c, ci] - X[:, a, ci])[:, None], sb, (X[:, c, ci] - X[:, b, ci])[:, None]], 1)
                rows.append(f); keys.append(c)
            P = N3.predict_proba(np.concatenate(rows))[:, 1].reshape(len(keys), R).T
            P = np.where(present[:, keys], P, 0)
            P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
            p3[:, a, b, keys] = P
    for k, (a, b, c) in enumerate(perms):
        tri[:, k] = (S[:, a, 1:] * p2[:, a, 1:, b]).sum(1) * p3[:, a, b, c]
    tri *= (present[:, [p[0] for p in perms]] & present[:, [p[1] for p in perms]] & present[:, [p[2] for p in perms]])
    return tri / np.maximum(tri.sum(1, keepdims=True), 1e-12), p2, p3


def scenario_probs(Sm, X, present):
    R = X.shape[0]
    q = Sm.predict_proba(X.reshape(R * 6, -1)).reshape(R, 6, -1)
    q[:, :, 0] = 0
    q = np.where(present[:, :, None], q, 0)
    return q / np.maximum(q.sum((1, 2), keepdims=True), 1e-12)


def metrics(tri, place, present):
    from research_eval import prob_metrics
    return prob_metrics(tri, place, present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--valid-from", default="2026-09-04")
    ap.add_argument("--valid-to", default="2026-11-03")
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    df, feats = load(Path(a.work))
    tr = df[df.date < a.valid_from]
    va = df[(df.date >= a.valid_from) & (df.date <= a.valid_to)]
    vc = T.venue_course_rates(tr)
    tr, va = T.fill_vc(tr, vc), T.fill_vc(va, vc)
    log(f"学習 {tr.race.nunique()} / 検証 {va.race.nunique()} レース")

    Sm = clf(a.iters).fit(tr[feats].to_numpy(float), winner_class(tr))
    log("S（1着×決まり手）学習")
    r2 = pair_rows(tr, feats, 2)
    N2 = clf(a.iters).fit(r2[pair_feats(feats, 2)].to_numpy(float), r2.y)
    log(f"N2（2着）学習 {len(r2)} 行")
    r3 = pair_rows(tr, feats, 3)
    N3 = clf(a.iters).fit(r3[pair_feats(feats, 3)].to_numpy(float), r3.y)
    log(f"N3（3着）学習 {len(r3)} 行")

    ids, present, place, X = race_tensor(va, feats)
    S = scenario_probs(Sm, X, present)
    tri, _, _ = tri_tenkai(S, N2, N3, X, present, feats)
    m = metrics(tri, place, present)
    log("展開モデル:", json.dumps({k: round(v, 4) for k, v in m.items()}))
    # 決まり手の当たり具合
    tech_true = va.drop_duplicates("race").set_index("race").tech.reindex(ids).to_numpy()
    pt = S.sum(1)[:, 1:]
    ok = tech_true > 0
    log("決まり手の予想 的中:", round(float((pt.argmax(1) + 1 == tech_true)[ok].mean()), 4),
        "平均の確率:", {TECH_NAMES[i + 1]: round(float(pt[:, i].mean()), 3) for i in range(5)},
        "実際:", {TECH_NAMES[i]: round(float((tech_true == i).mean()), 3) for i in range(1, 6)})
    np.save(Path(a.work) / "tenkai_valid_tri.npy", tri)
    pd.Series(ids).to_csv(Path(a.work) / "tenkai_valid_ids.csv", index=False)


if __name__ == "__main__":
    main()
