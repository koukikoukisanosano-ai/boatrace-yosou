"""学習（本番用）: 予想方法の比較で選ばれた「3つの平均」と、展示前の予想を学習して docs/model/model.json に書き出す

  1) 機械学習（勾配ブースティング）            … full
  2) 機械学習・新しい日ほど重く（半減期90日）    … full_rw
  3) 条件付きロジット（レース内で6艇を比べる）   … clogit
  4) 展示前（直前情報なし）の機械学習            … pre
  5) 展開モデル（1着×決まり手 → 2着 → 3着）       … tenkai   ← 直前情報ありの本番の予想
  6) 枠＋展示航走だけのモデル                      … ce       ← 展開モデルに重み exhibit_w で混ぜる（展示最重視）

検証: 最新日の60日前より前で学習 → 直近60日をページと同じ計算（Node）で予想し直して成績を出す
本番: 全期間で学習し直して書き出す

  python tools/train_all.py --work work
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from research import CLogit  # noqa: E402
from tenkai import COURSE, EXHIBIT, TECH, clf as tk_clf, pair_feats, pair_rows, winner_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"win": 1, "top2": 2, "top3": 3}
HALF_LIFE = 90
EXHIBIT_W = 0.85     # 展示航走の重み: 枠以外では展示航走がいちばん効く重さ（tools/tenkai_exhibit.py で決めた値）
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def load(path: Path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    meta = ["race", "date", "place", "preview"]
    feats = [c for c in df.columns if c not in meta]
    feats = [f for f in feats if f.startswith("vc") or not df[f].isna().all()]
    return df, feats


def weights(dates: pd.Series) -> np.ndarray:
    age = (pd.Timestamp(dates.max()) - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / HALF_LIFE)


def fit_gbm(tr, va, feats, recency=False, iters=1000):
    """検証つきで学習。最良の本数と、その本数で切った書き出しを返す"""
    vc = T.venue_course_rates(tr)
    trf, vaf = T.fill_vc(tr, vc), T.fill_vc(va, vc)
    X, Xv = trf[feats].to_numpy(float), vaf[feats].to_numpy(float)
    w = weights(trf.date) if recency else None
    best_iters, models = {}, {}
    for name, k in TARGETS.items():
        clf = T.make_clf(iters)
        clf.fit(X, (trf.place <= k).astype(int).to_numpy(), sample_weight=w)
        y_va = (vaf.place <= k).astype(int).to_numpy()
        best, best_ll = 0, 9e9
        for i, pr in enumerate(clf.staged_predict_proba(Xv), start=1):
            if i % 25:
                continue
            p = np.clip(pr[:, 1], 1e-9, 1 - 1e-9)
            ll = T.race_metrics(vaf, p)["logloss_winner"] if name == "win" else float(-(y_va * np.log(p) + (1 - y_va) * np.log(1 - p)).mean())
            if ll < best_ll:
                best, best_ll = i, ll
        best_iters[name] = best
        m = T.export_model(clf, len(feats))
        m["trees"] = m["trees"][:best]
        models[name] = m
        log(f"  {'新しい日重視 ' if recency else ''}{name}: 最良 {best} 本 ({best_ll:.4f})")
    return vc, models, best_iters


def refit_gbm(df, feats, best_iters, recency=False):
    vc = T.venue_course_rates(df)
    full = T.fill_vc(df, vc)
    X = full[feats].to_numpy(float)
    w = weights(full.date) if recency else None
    models = {}
    for name, k in TARGETS.items():
        n = max(50, int(best_iters[name] * 1.1))
        clf = T.make_clf(n)
        clf.fit(X, (full.place <= k).astype(int).to_numpy(), sample_weight=w)
        m = T.export_model(clf, len(feats))
        idx = np.random.RandomState(0).choice(len(X), 200, replace=False)
        diff = np.abs(T.raw_from_export(m, X[idx]) - clf.decision_function(X[idx])).max()
        if diff > 1e-4:
            raise RuntimeError(f"書き出したモデルが元と一致しません ({diff})")
        models[name] = m
        log(f"  {'新しい日重視 ' if recency else ''}{name}: {n} 本で全期間 (書き出し誤差 {diff:.1e})")
    return vc, models


def add_tech(df, races_path: Path):
    tech = {}
    for ln in open(races_path, encoding="utf-8"):
        r = json.loads(ln)
        tech[r["id"]] = TECH.get(r.get("tech") or 0, 0)
    df["tech"] = df.race.map(tech).fillna(0).astype(int)
    return df


def check(m, clf, X, multi=False):
    idx = np.random.RandomState(0).choice(len(X), min(200, len(X)), replace=False)
    got = T.raw_multi_from_export(m, X[idx]) if multi else T.raw_from_export(m, X[idx])
    diff = float(np.abs(got - clf.decision_function(X[idx])).max())
    if diff > 1e-4:
        raise RuntimeError(f"書き出したモデルが元と一致しません ({diff})")
    return diff


def fit_tenkai(df, feats):
    """展開モデル S・N2・N3 と、枠＋展示航走のモデル ce を学習して書き出しの形で返す"""
    vc = T.venue_course_rates(df)
    d = T.fill_vc(df, vc)
    X = d[feats].to_numpy(float)
    S = tk_clf().fit(X, winner_class(d))
    out = {"S": {"features": feats, **T.export_multiclass(S)}}
    e = check(out["S"], S, X, multi=True)
    for st in (2, 3):
        r = pair_rows(d, feats, st)
        pf = pair_feats(feats, st)
        Xp = r[pf].to_numpy(float)
        m = tk_clf().fit(Xp, r.y)
        out[f"N{st}"] = {"features": pf, **T.export_model(m, len(pf))}
        e = max(e, check(out[f"N{st}"], m, Xp))
    cef = [f for f in COURSE + EXHIBIT if f in feats]
    Xc = d[cef].to_numpy(float)
    ce = {"features": cef}
    for name, k in TARGETS.items():
        m = tk_clf().fit(Xc, (d.place <= k).astype(int).to_numpy())
        ce[name] = T.export_model(m, len(cef))
        e = max(e, check(ce[name], m, Xc))
    log(f"  展開モデル・展示モデル: 学習済み（書き出し誤差 {e:.1e}）")
    return out, ce


def export_clogit(cl: CLogit) -> dict:
    return {"features": list(cl.feats), "mu": [round(float(x), 9) for x in cl.mu], "sd": [round(float(x), 9) for x in cl.sd],
            "miss": [int(i) for i in cl.miss_cols], "w": [round(float(x), 9) for x in cl.w]}


def run_eval(model_path, valid_from, out_path, extra=()):
    r = subprocess.run(["node", str(ROOT / "tools" / "evaluate.mjs"), str(model_path), valid_from, "--json", str(out_path), *extra],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError("検証（ページと同じ計算）に失敗しました")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "model" / "model.json"))
    ap.add_argument("--valid-days", type=int, default=60)
    a = ap.parse_args()
    work = Path(a.work)
    df, feats = load(work / "full" / "features.tsv")
    df = add_tech(df, work / "full" / "races.jsonl")
    dp, feats_pre = load(work / "pre" / "features.tsv")
    valid_from = (pd.Timestamp(df.date.max()) - pd.Timedelta(days=a.valid_days)).strftime("%Y-%m-%d")
    tr, va = df[df.date < valid_from], df[df.date >= valid_from]
    trp, vap = dp[dp.date < valid_from], dp[dp.date >= valid_from]
    log(f"学習 {tr.race.nunique()} レース / 検証 {va.race.nunique()} レース（{valid_from}〜）")

    # ---------------- 検証
    vc, m_full, it_full = fit_gbm(tr, va, feats)
    _, m_rw, it_rw = fit_gbm(tr, va, feats, recency=True)
    cl = CLogit(feats).fit(tr, vc)
    log("  条件付きロジット: 学習済み")
    _, m_pre, it_pre = fit_gbm(trp, vap, feats_pre)
    valid_model = {
        "version": "valid", "trained_until": valid_from, "vc": vc, "ensemble": ["full", "full_rw", "clogit"],
        "full": {"features": feats, **m_full}, "full_rw": {"features": feats, **m_rw}, "clogit": export_clogit(cl),
        "pre": {"features": feats_pre, **m_pre},
    }
    vpath = work / "model_valid.json"
    vpath.write_text(json.dumps(valid_model, ensure_ascii=False, separators=(",", ":")))
    ev_ens = work / "eval_ens3.json"
    run_eval(vpath, valid_from, ev_ens)
    tk, ce = fit_tenkai(tr, feats)
    valid_model.update({"tenkai": tk, "ce": ce, "exhibit_w": EXHIBIT_W})
    vpath.write_text(json.dumps(valid_model, ensure_ascii=False, separators=(",", ":")))
    ev_path = work / "eval.json"
    run_eval(vpath, valid_from, ev_path)
    ev_tk0 = work / "eval_tenkai_w0.json"
    run_eval(vpath, valid_from, ev_tk0, ["--exhibit-w", "0"])

    # ---------------- 全期間で学習し直し
    vc_all, f_full = refit_gbm(df, feats, it_full)
    _, f_pre = refit_gbm(dp, feats_pre, it_pre)
    tk_all, ce_all = fit_tenkai(df, feats)
    out = {
        "version": time.strftime("%Y%m%d%H%M"), "trained_until": str(df.date.max()), "vc": vc_all,
        # 直前情報ありの予想は展開モデル（tenkai＋ce）。3つの平均は検証の比べ役なので、ページには機械学習 1つだけ（予備）を載せる
        "ensemble": ["full"],
        "full": {"features": feats, **f_full},
        "pre": {"features": feats_pre, **f_pre},
        "tenkai": tk_all, "ce": ce_all, "exhibit_w": EXHIBIT_W,
        "valid": {"full": {"valid_from": valid_from, "n_train_races": int(tr.race.nunique()), "n_valid_races": int(va.race.nunique()),
                           "best_iters": {"full": it_full, "full_rw": it_rw, "pre": it_pre}}},
        "blend_alpha": 0.5,
        "evaluation": json.loads(ev_path.read_text()),
        "evaluation_compare": {"ens3": json.loads(ev_ens.read_text()), "tenkai_w0": json.loads(ev_tk0.read_text())},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    log(f"書き出し {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
