"""学習（本番用）: 2連単の買い方の比較（research_exacta.py）で選ばれた予想の混ぜ方で学習して docs/model/model.json に書き出す

  直前情報ありの予想 = 次のモデルの重み付き掛け合わせ（mix。オッズは混ぜない）
    展開モデル（1着×決まり手 → 2着 → 3着）  … tenkai / tenkai_big
    機械学習（1着・2連対・3連対）            … full
    条件付きロジット（レース内で6艇を比べる）  … clogit
  枠＋展示航走だけのモデル                    … ce（「展示最重視」の表示で重み exhibit_w で混ぜる）
  展示前（直前情報なし）の機械学習            … pre

検証: 最新日の60日前より前で学習 → 直近60日をページと同じ計算（Node）で予想し直して成績を出す
本番: 全期間（2024-11〜）で学習し直して書き出す

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
TRAIN_FROM = "2024-11-01"   # Open API のデータは 2024-09-01 から
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


TENKAI_PARAMS = {"tenkai": (300, 0.08, 15), "tenkai_big": (600, 0.05, 31)}


def fit_tenkai(df, feats, name="tenkai"):
    """展開モデル S・N2・N3 を学習して書き出しの形で返す"""
    n, lr, leaves = TENKAI_PARAMS[name]
    mk = lambda: tk_clf(n, lr, leaves)  # noqa: E731
    vc = T.venue_course_rates(df)
    d = T.fill_vc(df, vc)
    X = d[feats].to_numpy(float)
    S = mk().fit(X, winner_class(d))
    out = {"S": {"features": feats, **T.export_multiclass(S)}}
    e = check(out["S"], S, X, multi=True)
    del X
    for st in (2, 3):
        r = pair_rows(d, feats, st)
        pf = pair_feats(feats, st)
        Xp, y = r[pf].to_numpy(float), r.y.to_numpy()
        del r
        m = mk().fit(Xp, y)
        out[f"N{st}"] = {"features": pf, **T.export_model(m, len(pf))}
        e = max(e, check(out[f"N{st}"], m, Xp))
        del Xp
    log(f"  展開モデル {name}: 学習済み（書き出し誤差 {e:.1e}）")
    return out


def fit_ce(df, feats):
    """枠＋展示航走だけのモデル"""
    vc = T.venue_course_rates(df)
    d = T.fill_vc(df, vc)
    cef = [f for f in COURSE + EXHIBIT if f in feats]
    Xc = d[cef].to_numpy(float)
    ce = {"features": cef}
    e = 0.0
    for name, k in TARGETS.items():
        m = tk_clf().fit(Xc, (d.place <= k).astype(int).to_numpy())
        ce[name] = T.export_model(m, len(cef))
        e = max(e, check(ce[name], m, Xc))
    log(f"  枠＋展示のモデル: 学習済み（書き出し誤差 {e:.1e}）")
    return ce


def load_bet():
    """2連単の買い方（research_exacta.py の結果）。予想の混ぜ方・確率の補正・買い目の条件"""
    p = ROOT / "docs" / "data" / "research_exacta.json"
    if not p.exists():
        return {"tenkai": 1 / 3, "full": 1 / 3, "clogit": 1 / 3}, None
    R = json.loads(p.read_text())
    b = R["best"]
    mix = {("full" if k == "gbm" else k): v for k, v in R["combos"][b["prob"]].items()}
    # 通知する時点（締切約5分前）の実際のオッズで十分に検証できたら（tools/odds_drift.py）、その買い方を使う
    dp = ROOT / "docs" / "data" / "drift.json"
    if dp.exists():
        Dr = json.loads(dp.read_text())
        if Dr.get("adopt") and Dr.get("best_early"):
            bt = Dr["best_early"]
            log("締切約5分前のオッズで選んだ買い方を使います:", bt["rule"], bt["roi"], bt["bets"])
            b = {**b, "rule": {**bt["rule"], "edge": 1.0}, "label": "2連単（締切約5分前のオッズで検証）", "dev": bt, "hold": bt}
    return mix, {"kind": "exacta", "prob": b["prob"], "gamma": b["gamma"], "rule": {**b["rule"], "gamma": b["gamma"], "alpha": 1},
                 "label": b["label"], "dev": b["dev"], "hold": b["hold"], "hold_months": b.get("hold_months"), "when": R.get("when"),
                 "dev_months": R.get("dev"), "hold_period": R.get("hold")}


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
    # 近況がたまるまでの最初の2か月は学習に使わない
    df, dp = df[df.date >= TRAIN_FROM], dp[dp.date >= TRAIN_FROM]
    valid_from = (pd.Timestamp(df.date.max()) - pd.Timedelta(days=a.valid_days)).strftime("%Y-%m-%d")
    tr, va = df[df.date < valid_from], df[df.date >= valid_from]
    trp, vap = dp[dp.date < valid_from], dp[dp.date >= valid_from]
    log(f"学習 {tr.race.nunique()} レース / 検証 {va.race.nunique()} レース（{valid_from}〜）")

    mix, bet = load_bet()
    tk_names = [k for k in mix if k.startswith("tenkai")] or ["tenkai"]
    log("予想の混ぜ方:", mix, "買い方:", bet and bet["label"])

    # ---------------- 検証（直近 valid_days 日を、それより前だけで学習したモデルで予想し直す）
    use_full = "full" in mix
    vc = T.venue_course_rates(tr)
    _, m_pre, it_pre = fit_gbm(trp, vap, feats_pre)
    valid_model = {"version": "valid", "trained_until": valid_from, "vc": vc, "mix": mix, "blend_alpha": 1,
                   "pre": {"features": feats_pre, **m_pre}, "ce": fit_ce(tr, feats), "exhibit_w": EXHIBIT_W}
    if use_full:
        vc, m_full, it_full = fit_gbm(tr, va, feats)
        valid_model.update({"full": {"features": feats, **m_full}, "ensemble": ["full"]})
    if "clogit" in mix:
        valid_model["clogit"] = export_clogit(CLogit(feats).fit(tr, vc))
    for k in tk_names:
        valid_model[k] = fit_tenkai(tr, feats, k)
    vpath = work / "model_valid.json"
    vpath.write_text(json.dumps(valid_model, ensure_ascii=False, separators=(",", ":")))
    ev_path = work / "eval.json"
    run_eval(vpath, valid_from, ev_path)
    ev_ex = work / "eval_exhibit.json"
    run_eval(vpath, valid_from, ev_ex, ["--exhibit-w", str(EXHIBIT_W)])
    del valid_model

    # ---------------- 全期間で学習し直し
    vc_all = T.venue_course_rates(df)
    _, f_pre = refit_gbm(dp, feats_pre, it_pre)
    out = {
        "version": time.strftime("%Y%m%d%H%M"), "trained_until": str(df.date.max()), "vc": vc_all,
        # 直前情報ありの予想: mix のモデルを掛け合わせる（オッズは混ぜない）。展示最重視の表示では ce を exhibit_w で混ぜる
        "mix": mix, "bet": bet, "blend_alpha": 1,
        "pre": {"features": feats_pre, **f_pre},
        "ce": fit_ce(df, feats), "exhibit_w": EXHIBIT_W,
        "valid": {"full": {"valid_from": valid_from, "n_train_races": int(tr.race.nunique()), "n_valid_races": int(va.race.nunique()),
                           "best_iters": {"pre": it_pre}}},
        "evaluation": json.loads(ev_path.read_text()),
        "evaluation_compare": {"exhibit": json.loads(ev_ex.read_text())},
    }
    if use_full:
        _, f_full = refit_gbm(df, feats, it_full)
        out.update({"full": {"features": feats, **f_full}, "ensemble": ["full"]})
    if "clogit" in mix:
        out["clogit"] = export_clogit(CLogit(feats).fit(df, vc_all))
    for k in tk_names:
        out[k] = fit_tenkai(df, feats, k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    log(f"書き出し {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
