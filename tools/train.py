"""学習: work/features.tsv → docs/model/model.json（スマホのブラウザで動かせる形の勾配ブースティング木）

  python tools/train.py [--work work] [--valid-from 2026-09-01] [--final]

--final を付けると、検証で決めた木の本数で全期間を学習し直して書き出す。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"win": 1, "top2": 2, "top3": 3}
PRIOR_K = 50.0


def venue_course_rates(df: pd.DataFrame) -> dict:
    """場×コース（スタート展示の進入）ごとの 1着・2連対・3連対率。少ない場は全体へならす。"""
    out: dict = {"0": {}}
    g = df.assign(w=(df.place == 1).astype(float), t2=(df.place <= 2).astype(float), t3=(df.place <= 3).astype(float))
    allc = g.groupby("course")[["w", "t2", "t3"]].mean()
    for c, r in allc.iterrows():
        out["0"][str(int(c))] = [round(float(r.w), 6), round(float(r.t2), 6), round(float(r.t3), 6)]
    agg = g.groupby(["jcd", "course"])[["w", "t2", "t3"]].agg(["sum", "count"])
    for (j, c), r in agg.iterrows():
        n = r[("w", "count")]
        pri = out["0"][str(int(c))]
        vals = [(r[(k, "sum")] + PRIOR_K * pri[i]) / (n + PRIOR_K) for i, k in enumerate(["w", "t2", "t3"])]
        out.setdefault(str(int(j)), {})[str(int(c))] = [round(float(v), 6) for v in vals]
    return out


def fill_vc(df: pd.DataFrame, vc: dict) -> pd.DataFrame:
    def look(j, c, i):
        v = vc.get(str(int(j)), {}).get(str(int(c))) or vc["0"].get(str(int(c)))
        return v[i] if v else np.nan
    keys = list(zip(df.jcd, df.course))
    cache = {k: [look(k[0], k[1], i) for i in range(3)] for k in set(keys)}
    arr = np.array([cache[k] for k in keys])
    df = df.copy()
    df["vcWin"], df["vcTop2"], df["vcTop3"] = arr[:, 0], arr[:, 1], arr[:, 2]
    return df


def make_clf(n_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=n_iter, max_leaf_nodes=31, min_samples_leaf=60,
        l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=0,
    )


def export_model(clf: HistGradientBoostingClassifier, n_features: int) -> dict:
    base = float(np.ravel(clf._baseline_prediction)[0])
    trees = []
    for it in clf._predictors:
        nodes = it[0].nodes
        if np.any(nodes["is_categorical"]):
            raise RuntimeError("カテゴリ分割は未対応")
        leaf = nodes["is_leaf"].astype(bool)
        fe = nodes["feature_idx"].astype(int).tolist()
        # しきい値は丸めない（データの値と同じ値がしきい値になるため、丸めると分かれ方が変わる）
        th = [1e300 if x == math.inf else (-1e300 if x == -math.inf else float(x)) for x in nodes["num_threshold"].tolist()]
        le = np.where(leaf, 0, nodes["left"]).astype(int).tolist()
        ri = np.where(leaf, 0, nodes["right"]).astype(int).tolist()
        ml = nodes["missing_go_to_left"].astype(int).tolist()
        va = [float(f"{x:.7g}") for x in nodes["value"].tolist()]
        for i in range(len(fe)):
            if leaf[i]:
                fe[i], th[i], ml[i] = 0, 0, 0
            else:
                va[i] = 0
            if th[i] == int(th[i]) and abs(th[i]) < 1e15:
                th[i] = int(th[i])
        trees.append([fe, th, le, ri, ml, va])
    return {"base": base, "trees": trees}


def raw_from_export(m: dict, X: np.ndarray) -> np.ndarray:
    out = np.full(len(X), m["base"])
    for fe, th, le, ri, ml, va in m["trees"]:
        for r in range(len(X)):
            i = 0
            while le[i] != 0:
                v = X[r, fe[i]]
                if v != v:
                    i = le[i] if ml[i] else ri[i]
                else:
                    i = le[i] if v <= th[i] else ri[i]
            out[r] += va[i]
    return out


def race_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    d = df[["race", "place"]].copy()
    d["p"] = p
    d["pn"] = d.p / d.groupby("race").p.transform("sum")
    win = d[d.place == 1]
    ll = float(-np.log(np.clip(win.pn, 1e-9, 1)).mean())
    top = d.loc[d.groupby("race").pn.idxmax()]
    acc = float((top.place == 1).mean())
    return {"races": int(d.race.nunique()), "logloss_winner": round(ll, 4), "top1_acc": round(acc, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--valid-from", default="", help="検証に使う期間の開始日（空なら最新日の60日前）")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--features", default="", help="使う特徴量をカンマ区切りで（比較用）")
    ap.add_argument("--drop", default="", help="除く特徴量をカンマ区切りで（比較用）")
    ap.add_argument("--out", default=str(ROOT / "docs" / "model" / "model.json"))
    ap.add_argument("--slot", default="full", choices=["full", "pre"], help="full=直前情報あり / pre=展示前")
    a = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(Path(a.work) / "features.tsv", sep="\t", low_memory=False)
    meta = ["race", "date", "place", "preview"]
    all_feats = [c for c in df.columns if c not in meta]
    feats = all_feats
    if a.features:
        feats = [f for f in a.features.split(",") if f]
    if a.drop:
        dropped = set(a.drop.split(","))
        feats = [f for f in feats if f not in dropped]
    empty = [f for f in feats if df[f].isna().all() and not f.startswith("vc")]
    if empty:
        print("値が無いので除く特徴:", ",".join(empty))
        feats = [f for f in feats if f not in empty]
    print(f"読み込み {len(df)} 行 / {df.race.nunique()} レース / 特徴 {len(feats)} 個 ({time.time() - t0:.1f}s)")

    if not a.valid_from:
        last = pd.Timestamp(df.date.max())
        a.valid_from = (last - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    tr = df[df.date < a.valid_from]
    va = df[df.date >= a.valid_from]
    vc = venue_course_rates(tr)
    tr, va = fill_vc(tr, vc), fill_vc(va, vc)
    result = {"features": feats, "valid_from": a.valid_from, "n_train_races": int(tr.race.nunique()), "n_valid_races": int(va.race.nunique())}
    best_iters = {}
    valid_models = {}
    for name, k in TARGETS.items():
        y_tr = (tr.place <= k).astype(int).to_numpy()
        y_va = (va.place <= k).astype(int).to_numpy()
        clf = make_clf(a.iters)
        clf.fit(tr[feats].to_numpy(dtype=float), y_tr)
        Xv = va[feats].to_numpy(dtype=float)
        best, best_ll, curve = 0, 9e9, []
        for i, pr in enumerate(clf.staged_predict_proba(Xv), start=1):
            if i % 25 == 0 or i == a.iters:
                p = np.clip(pr[:, 1], 1e-9, 1 - 1e-9)
                ll = float(-(y_va * np.log(p) + (1 - y_va) * np.log(1 - p)).mean())
                curve.append((i, round(ll, 5)))
                if ll < best_ll:
                    best, best_ll = i, ll
        best_iters[name] = best
        p_best = None
        for i, pr in enumerate(clf.staged_predict_proba(Xv), start=1):
            if i == best:
                p_best = pr[:, 1]
                break
        msg = f"[{name}] 最良 {best} 本 logloss={best_ll:.5f}"
        if name == "win":
            rm = race_metrics(va, p_best)
            result["valid_win"] = rm
            msg += f" / 1着予想の的中率 {rm['top1_acc']:.3f} / 勝者のlogloss {rm['logloss_winner']:.4f}"
        print(msg, f"({time.time() - t0:.0f}s)")
        result[f"curve_{name}"] = curve
        if not a.final:
            m = export_model(clf, len(feats))
            m["trees"] = m["trees"][:best]
            valid_models[name] = m
    result["best_iters"] = best_iters

    if not a.final:
        Path(a.work, "train_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        mv = {"version": "valid", "trained_until": a.valid_from, "vc": vc, a.slot: {"features": feats, **valid_models}, "valid": result}
        Path(a.work, f"model_valid_{a.slot}.json").write_text(json.dumps(mv, ensure_ascii=False, separators=(",", ":")))
        print("検証用モデルを書き出し:", Path(a.work, f"model_valid_{a.slot}.json"))
        return

    # 全期間で学習し直して書き出す
    full = fill_vc(df, venue_course_rates(df))
    vc_full = venue_course_rates(df)
    models = {}
    X = full[feats].to_numpy(dtype=float)
    for name, k in TARGETS.items():
        n_iter = max(50, int(best_iters[name] * 1.1))
        clf = make_clf(n_iter)
        clf.fit(X, (full.place <= k).astype(int).to_numpy())
        m = export_model(clf, len(feats))
        # 書き出した木で計算し直して、元のモデルと一致するか確認
        idx = np.random.RandomState(0).choice(len(X), 300, replace=False)
        diff = np.abs(raw_from_export(m, X[idx]) - clf.decision_function(X[idx])).max()
        print(f"[{name}] {n_iter} 本で全期間を学習 / 書き出しの誤差 {diff:.2e}")
        if diff > 1e-4:
            raise RuntimeError("書き出したモデルが元と一致しません")
        models[name] = m
    out = {}
    if Path(a.out).exists():
        out = json.loads(Path(a.out).read_text())
    out.update({"version": time.strftime("%Y%m%d%H%M"), "trained_until": str(df.date.max()), "vc": vc_full})
    out[a.slot] = {"features": feats, **models}
    out.setdefault("valid", {})[a.slot] = result
    ev = Path(a.work) / "eval.json"
    if a.slot == "full" and ev.exists():
        out["evaluation"] = json.loads(ev.read_text())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"書き出し {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
