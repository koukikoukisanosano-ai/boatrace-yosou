"""2連単で回収率100%超えを狙う: オッズを混ぜない予想（モデルだけ）を作り込み、期待値の買い方を比べる

  1) 予想し直し（重い）:  python tools/research_exacta.py fit  --work work/full
       開発期間の前日までで学習 → 開発期間を予想、本番テスト期間の前日までで学習 → 本番テスト期間を予想
       モデル: 展開モデル（標準・大きめ）、機械学習（1着・2連対・3連対）、条件付きロジット、枠＋展示のモデル
  2) 採点（軽い）:        python tools/research_exacta.py eval --work work/full
       予想の組み合わせ × 確率の補正 × 買い方（期待値・確率・倍率・点数・市場との差）を開発期間で選び、本番テスト期間で確かめる
       オッズは「確率」には混ぜない。オッズは値段（期待値の計算）にだけ使う
出力: work/full/exacta_preds.pkl, docs/data/research_exacta.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from research import EXA, CLogit, fit_gbm, load_odds, normalize_place, pl_from_strength, predict_gbm, tri_iter  # noqa: E402
from research_eval import bet_probs, odds_matrix, payout_matrix, score  # noqa: E402
from tenkai import COURSE, EXHIBIT, TECH, clf, pair_feats, pair_rows, race_tensor, scenario_probs, tri_tenkai, winner_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def load(work: Path):
    df = pd.read_csv(work / "features.tsv", sep="\t", low_memory=False)
    meta = ["race", "date", "place", "preview"]
    feats = [c for c in df.columns if c not in meta]
    feats = [f for f in feats if f.startswith("vc") or not df[f].isna().all()]
    tech = {}
    for ln in open(work / "races.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        tech[r["id"]] = TECH.get(r.get("tech") or 0, 0)
    df["tech"] = df.race.map(tech).fillna(0).astype(int)
    return df, feats


# ---------------------------------------------------------------- 1) 予想し直し
def fit_predict(tr, te, feats, variants):
    vc = T.venue_course_rates(tr)
    trf, tef = T.fill_vc(tr, vc), T.fill_vc(te, vc)
    ids, present, place, X = race_tensor(tef, feats)
    R = len(ids)
    out = {"ids": ids, "present": present, "place": place}
    for name, (n, lr, leaves) in variants.items():
        mk = lambda: clf(n, lr, leaves)  # noqa: E731
        Sm = mk().fit(trf[feats].to_numpy(float), winner_class(trf))
        r2 = pair_rows(trf, feats, 2); N2 = mk().fit(r2[pair_feats(feats, 2)].to_numpy(float), r2.y); del r2
        r3 = pair_rows(trf, feats, 3); N3 = mk().fit(r3[pair_feats(feats, 3)].to_numpy(float), r3.y); del r3
        out[name], _, _ = tri_tenkai(scenario_probs(Sm, X, present), N2, N3, X, present, feats)
        log(f"  展開モデル {name}")
    # 枠＋展示航走だけ
    cef = [f for f in COURSE + EXHIBIT if f in feats]
    Xc = X[:, :, [feats.index(f) for f in cef]].reshape(R * 6, -1)
    p = {k: clf().fit(trf[cef].to_numpy(float), (trf.place <= m).astype(int)).predict_proba(Xc)[:, 1].reshape(R, 6)
         for k, m in (("win", 1), ("top2", 2), ("top3", 3))}
    out["ce"] = tri_iter(*normalize_place(p["win"], p["top2"], p["top3"], present), present)
    log("  枠＋展示")
    # 機械学習（1着・2連対・3連対）
    _, models = fit_gbm(tr, feats)
    Xf = X.reshape(R * 6, -1)
    pg = {k: m.predict_proba(Xf)[:, 1].reshape(R, 6) for k, m in models.items()}
    out["gbm"] = tri_iter(*normalize_place(pg["win"], pg["top2"], pg["top3"], present), present)
    log("  機械学習")
    # 条件付きロジット
    cl = CLogit(feats).fit(tr, vc)
    s = np.zeros((R, 6))
    idx = {r: i for i, r in enumerate(ids)}
    s[tef.race.map(idx).to_numpy(), tef.waku.to_numpy().astype(int) - 1] = cl.strength(te, vc)
    out["clogit"] = pl_from_strength(s, present)[1]
    log("  条件付きロジット")
    return out


def cmd_fit(a):
    work = Path(a.work)
    df, feats = load(work)
    df = df[df.date >= a.train_from]
    log(f"{df.race.nunique()} レース（{df.date.min()}〜{df.date.max()}）・特徴量 {len(feats)}")
    if not a.dev_from or not a.hold_from:   # 本番テスト = 終わった直近4か月、開発 = その前の4か月
        months = sorted({d[:7] for d in df.date})[:-1]
        a.hold_from = a.hold_from or f"{months[-4]}-01"
        a.dev_from = a.dev_from or f"{months[-8]}-01"
    variants = {"tenkai": (300, 0.08, 15)}
    if a.big:
        variants["tenkai_big"] = (600, 0.05, 31)
    out = {}
    for name, start, end in (("dev", a.dev_from, a.hold_from), ("hold", a.hold_from, "2099-12-31")):
        tr = df[df.date < start]
        te = df[(df.date >= start) & (df.date < end)]
        te = te[te.date.str[:7] < df.date.max()[:7]] if name == "hold" else te   # 途中の月は使わない
        log(f"{name}: 学習 {tr.race.nunique()} レース（〜{tr.date.max()}）→ 予想 {te.race.nunique()} レース（{te.date.min()}〜{te.date.max()}）")
        out[name] = fit_predict(tr, te, feats, variants)
        pickle.dump(out, open(work / "exacta_preds.pkl", "wb"))
    log("保存", work / "exacta_preds.pkl")


# ---------------------------------------------------------------- 2) 採点
def geo(parts, ws):
    lg = sum(w * np.log(np.maximum(p, 1e-15)) for p, w in zip(parts, ws))
    t = np.exp(lg - lg.max(1, keepdims=True))
    return t / t.sum(1, keepdims=True)


def power(p, g):
    q = np.maximum(p, 1e-15) ** g
    return q / q.sum(1, keepdims=True)


def exacta_truth(place, present):
    order = np.argsort(np.where(present, place, 99), 1)[:, :2]
    ok = (np.take_along_axis(place, order, 1) == [1, 2]).all(1)
    return np.array([EXA.index((a, b)) for a, b in order]), ok


EV_T = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
MINP = [0.0, 0.02, 0.04, 0.07, 0.1]
MAXO = [15.0, 30.0, 60.0, 100.0]
MAXK = [1, 2, 3, 5]
EDGE = [1.0, 1.2, 1.4]       # 予想の確率 ÷ 市場の確率（オッズの逆数） の下限


def rules(prob, odds, valid):
    o = np.where(np.isfinite(odds) & (odds > 0), odds, np.nan)
    ev = prob * np.nan_to_num(o, nan=0)
    inv = np.where(np.isfinite(o), 1 / o, 0)
    mkt = inv / np.maximum(inv.sum(1, keepdims=True), 1e-12)
    edge = prob / np.maximum(mkt, 1e-12)
    base = valid[:, None] & np.isfinite(o)
    rank = np.argsort(np.argsort(-ev, 1), 1)
    for t, mp, mo, k, ed in itertools.product(EV_T, MINP, MAXO, MAXK, EDGE):
        if ed > 1.0 and t > 1.2:
            continue
        yield {"t": t, "minp": mp, "maxo": mo, "k": k, "edge": ed}, base & (ev >= t) & (prob >= mp) & (np.nan_to_num(o, nan=1e9) <= mo) & (rank < k) & (edge >= ed)


def label(r):
    s = f"2連単 期待値{r['t']}以上"
    if r["minp"]:
        s += f"・確率{r['minp'] * 100:g}%以上"
    if r.get("edge", 1) > 1:
        s += f"・市場の{r['edge']}倍以上の確率"
    return s + f"・{int(r['maxo'])}倍以下・1レース{r['k']}点まで"


def cmd_eval(a):
    work = Path(a.work)
    P = pickle.load(open(work / "exacta_preds.pkl", "rb"))
    races = {}
    need = set(np.concatenate([P[s]["ids"] for s in P]))
    for ln in open(work / "races.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        if r["id"] in need:
            races[r["id"]] = r
    D = {}
    for s, b in P.items():
        ids = b["ids"]
        odds = load_odds(sorted({f"{i[:4]}-{i[4:6]}-{i[6:8]}" for i in ids}))
        eo = odds_matrix(ids, odds, "exacta")
        truth, ok = exacta_truth(b["place"], b["present"])
        has = np.isfinite(eo).sum(1) >= 20
        D[s] = {**b, "eo": eo, "pay": payout_matrix(ids, races, "exacta"), "truth": truth, "ok": ok & has, "has": has,
                "months": np.array([i[:4] + "-" + i[4:6] for i in ids])}
        log(f"{s}: {len(ids)} レース・オッズあり {has.sum()}")
    base_names = [k for k in P["dev"] if k not in ("ids", "present", "place")]
    # 予想の組み合わせ（確率はモデルだけ）
    combos = {n: {n: 1.0} for n in base_names}
    tk = "tenkai_big" if "tenkai_big" in base_names else "tenkai"
    combos.update({
        "展開＋機械学習＋ロジット": {tk: 1 / 3, "gbm": 1 / 3, "clogit": 1 / 3},
        "展開＋機械学習": {tk: 0.5, "gbm": 0.5},
        "展開2つ＋機械学習＋ロジット": {"tenkai": 0.25, tk: 0.25, "gbm": 0.25, "clogit": 0.25},
        "展開＋機械学習＋ロジット＋展示0.3": {tk: 0.7 / 3, "gbm": 0.7 / 3, "clogit": 0.7 / 3, "ce": 0.3},
        "展開（標準＋大きめ）": {"tenkai": 0.5, tk: 0.5},
        "展開（大きめ）＋機械学習0.25": {tk: 0.75, "gbm": 0.25},
        "展開（大きめ）＋展示0.15": {tk: 0.85, "ce": 0.15},
    })

    def exa_of(s, mix, g=1.0):
        t = geo([D[s][k] for k in mix], list(mix.values()))
        e = bet_probs(t)["exacta"]
        return power(e, g) if g != 1.0 else e

    rows_prob, cand = [], []
    for name, mix in combos.items():
        if any(k not in D["dev"] for k in mix):
            continue
        for g in (0.8, 0.9, 1.0, 1.1, 1.2):
            row = {"name": name, "gamma": g}
            for s in D:
                e = exa_of(s, mix, g); d = D[s]; R = np.arange(len(e))
                ll = -np.log(np.clip(e[R, d["truth"]], 1e-12, 1))
                row[f"{s}_ll"] = float(ll[d["ok"]].mean())
                row[f"{s}_top1"] = float((np.argmax(e, 1) == d["truth"])[d["ok"]].mean())
            rows_prob.append(row)
    # 市場（確定オッズ）の当たり具合（比べるだけ）
    mrow = {"name": "市場（確定オッズ）", "gamma": 1.0}
    for s, d in D.items():
        inv = np.where(np.isfinite(d["eo"]) & (d["eo"] > 0), 1 / d["eo"], 0); mk = inv / np.maximum(inv.sum(1, keepdims=True), 1e-12)
        R = np.arange(len(mk))
        mrow[f"{s}_ll"] = float((-np.log(np.clip(mk[R, d["truth"]], 1e-12, 1)))[d["ok"]].mean())
        mrow[f"{s}_top1"] = float((np.argmax(mk, 1) == d["truth"])[d["ok"]].mean())
    for r in sorted(rows_prob, key=lambda r: r["dev_ll"])[:12] + [mrow]:
        log(f"{r['name']} γ{r['gamma']}: 開発 logloss {r['dev_ll']:.4f} 1点 {r['dev_top1']:.3f} / 本番 {r['hold_ll']:.4f} {r['hold_top1']:.3f}")

    # 買い方: 確率の当たり具合が良い上位の予想で総当たり
    best_prob = sorted(rows_prob, key=lambda r: r["dev_ll"])[: a.top_probs]
    names = {(r["name"], r["gamma"]) for r in best_prob} | {(n, 1.0) for n in combos}
    for name, g in sorted(names):
        e = {s: exa_of(s, combos[name], g) for s in D}
        for (rule, md), (_, mh) in zip(rules(e["dev"], D["dev"]["eo"], D["dev"]["has"]), rules(e["hold"], D["hold"]["eo"], D["hold"]["has"])):
            sd = score(md, D["dev"]["pay"])
            if sd["bets"] < a.min_bets:
                continue
            sh = score(mh, D["hold"]["pay"])
            # 月ごとの回収率（本番テスト）
            mon = {}
            for m in sorted(set(D["hold"]["months"])):
                sel = D["hold"]["months"] == m
                mon[str(m)] = round(float(D["hold"]["pay"][sel][mh[sel]].sum() / max(mh[sel].sum(), 1) / 100), 3)
            cand.append({"prob": name, "gamma": g, "rule": rule, "label": label(rule), "dev": sd, "hold": sh, "hold_months": mon})
        log(f"{name} γ{g}: 買い方 {len(cand)} 通り")
    cand.sort(key=lambda c: -c["dev"]["lb"])
    best = cand[0]
    log("開発期間で一番:", best["prob"], best["gamma"], best["label"], "開発", round(best["dev"]["roi"], 3), "±", round(best["dev"]["se"], 3),
        best["dev"]["bets"], "→ 本番", round(best["hold"]["roi"], 3), "±", round(best["hold"]["se"], 3), best["hold"]["bets"], best["hold_months"])
    for c in cand[:15]:
        log(f"  {c['prob']} γ{c['gamma']} {c['label']}: 開発 {c['dev']['roi']:.3f}±{c['dev']['se']:.3f}({c['dev']['bets']}) → 本番 {c['hold']['roi']:.3f}±{c['hold']['se']:.3f}({c['hold']['bets']}) {c['hold_months']}")
    out = {"when": time.strftime("%Y-%m-%d"), "dev": sorted(set(D["dev"]["months"])), "hold": sorted(set(D["hold"]["months"])),
           "hold_days": int(len({i[:8] for i in D["hold"]["ids"]})), "prob_table": sorted(rows_prob, key=lambda r: r["dev_ll"])[:20] + [mrow],
           "best": best, "top": cand[:40], "n": len(cand), "combos": combos}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, default=float))
    (work / "exacta_cands.json").write_text(json.dumps(cand, ensure_ascii=False, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-from", default="2024-11-01", help="学習に使う最初の日（近況がたまるまでの最初の2か月は使わない）")
    ap.add_argument("--dev-from", default="", help="開発期間の最初の日（空なら本番テストの4か月前）")
    ap.add_argument("--hold-from", default="", help="本番テスト期間の最初の日（空なら終わった直近4か月）")
    ap.add_argument("--big", action="store_true", help="大きめの展開モデルも試す")
    ap.add_argument("--min-bets", type=int, default=2000)
    ap.add_argument("--top-probs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "research_exacta.json"))
    a = ap.parse_args()
    cmd_fit(a) if a.cmd == "fit" else cmd_eval(a)


if __name__ == "__main__":
    main()
