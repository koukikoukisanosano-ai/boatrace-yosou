"""選手と競艇場の相性: 今の予想（公平な検証の、その月より前だけで学習した予想）に足して、当たり具合が良くなるかを確かめる。

今の予想には、すでに「当地勝率」（出走表）と「その場での近況の1着率」（pVenueWin）が入っている。
ここでは、それより長い期間（2016年から）の、選手ごとの
  A1: その場での1着率 − その選手の全体の1着率（場との相性そのもの）
  A2: その場・そのコースでの1着率 − その選手のそのコースの全体の1着率
  A3: 全体の平均着順 − その場での平均着順（＋なら、その場ではいつもより良い）
を作り、予想の確率に「log p + β・相性」の形で足して、2024-11〜2025-10 で β を決め、2025-11〜2026-10 で確かめる。

  python tools/affinity_test.py --mbrace data/mbrace --preds fair --out affinity.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eval import bet_probs  # noqa: E402
from research_exacta import geo, log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIX = {"tenkai_big": 0.5, "gbm": 0.5}   # 本番の予想（展開＋機械学習）


def day_files(mbrace: Path):
    oa = sorted((ROOT / "data" / "openapi").rglob("*.json.gz"))
    first = oa[0].name
    mb = sorted(p for p in mbrace.rglob("*.json.gz") if p.name < first) if mbrace.exists() else []
    return mb + oa


def races_of(p: Path):
    j = json.loads(gzip.decompress(p.read_bytes()))
    st = (j.get("programs") or {}).get("stadiums") or {}
    for jcd, v in st.items():
        for rno, rc in (v.get("races") or {}).items():
            res = rc.get("result") or {}
            rr = res.get("racers") or {}
            if not rr:
                continue
            boats = []
            for w, x in rr.items():
                num = x.get("number")
                if not num:
                    continue
                pl = x.get("place_number")
                boats.append((int(w), int(num), x.get("course_number") or int(w), pl if isinstance(pl, int) and 1 <= pl <= 6 else None))
            if len(boats) >= 5:
                yield f"{p.name[:8]}-{int(jcd)}-{int(rno)}", int(jcd), boats


def shrink(k, n, prior, m):
    return (k + prior * m) / (n + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbrace", default="data/mbrace")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--from", dest="start", default="2024-11")
    ap.add_argument("--split", default="2025-11", help="この月から確かめる（それより前で β を決める）")
    ap.add_argument("--out", default="affinity.json")
    a = ap.parse_args()

    # 1) 2016年からの成績を日付順に積み上げ、試す月のレースだけ「その日より前」の相性を記録する
    tot = defaultdict(lambda: [0, 0, 0.0])        # 選手 → [走った数, 1着, 着順の合計]
    ven = defaultdict(lambda: [0, 0, 0.0])        # (選手, 場)
    crs = defaultdict(lambda: [0, 0])             # (選手, コース)
    vc = defaultdict(lambda: [0, 0])              # (選手, 場, コース)
    feat = {}
    start = a.start.replace("-", "")
    files = day_files(Path(a.mbrace))
    log(f"日別ファイル {len(files)}（{files[0].name}〜{files[-1].name}）")
    for p in files:
        day = list(races_of(p))
        if p.name[:6] >= start:
            for rid, jcd, boats in day:
                f = {}
                for w, num, c, pl in boats:
                    t, v, cc, x = tot[num], ven[(num, jcd)], crs[(num, c)], vc[(num, jcd, c)]
                    w_all = shrink(t[1], t[0], 1 / 6, 30)
                    w_v = shrink(v[1], v[0], w_all, 20)
                    w_c = shrink(cc[1], cc[0], 1 / 6, 20)
                    w_vc = shrink(x[1], x[0], w_c, 10)
                    fa = t[2] / t[0] if t[0] else 3.5
                    fv = (v[2] + fa * 10) / (v[0] + 10)
                    f[w] = (w_v - w_all, w_vc - w_c, fa - fv, v[0])
                feat[rid] = f
        for rid, jcd, boats in day:            # その日の結果は日が終わってから足す
            for w, num, c, pl in boats:
                fin = pl if pl else 6
                for d in (tot[num], ven[(num, jcd)]):
                    d[0] += 1; d[1] += pl == 1; d[2] += fin
                for d in (crs[(num, c)], vc[(num, jcd, c)]):
                    d[0] += 1; d[1] += pl == 1
    log(f"相性を作ったレース {len(feat)}")

    # 2) 公平な検証の予想（1着の確率）と合わせる
    rows = []   # (月, レース番号, 枠, log p, A1, A2, A3, 場での出走数, 1着か)
    import pickle
    for fi, f in enumerate(sorted(Path(a.preds).rglob("fair_*.pkl"))):
        b = pickle.load(open(f, "rb"))
        if b["month"] < a.start:
            continue
        win = bet_probs(geo([b[k].astype(float) for k in MIX], list(MIX.values())))["win"]
        for i, rid in enumerate(b["ids"]):
            fr = feat.get(str(rid))
            if not fr:
                continue
            place = b["place"][i]
            for w in range(1, 7):
                if w not in fr or not b["present"][i][w - 1]:
                    continue
                pw = max(win[i, w - 1], 1e-6)
                rows.append((b["month"], fi * 1_000_000 + i, w, np.log(pw), *fr[w], float(place[w - 1] == 1)))
    R = np.array([r[1:] for r in rows], dtype=float)
    mon = np.array([r[0] for r in rows])
    log(f"艇の行 {len(R)}")
    race = R[:, 0]; lp = R[:, 2]; X = R[:, 3:6]; nv = R[:, 6]; y = R[:, 7]
    X = (X - X[mon < a.split].mean(0)) / X[mon < a.split].std(0)

    def nll(beta, m):
        s = lp[m] + X[m] @ beta
        rr = race[m]
        _, inv = np.unique(rr, return_inverse=True)
        mx = np.full(inv.max() + 1, -np.inf); np.maximum.at(mx, inv, s)
        e = np.exp(s - mx[inv]); z = np.zeros(inv.max() + 1); np.add.at(z, inv, e)
        lpn = s - mx[inv] - np.log(z[inv])
        return -(lpn * y[m]).sum() / y[m].sum()

    tr, te = mon < a.split, mon >= a.split
    from scipy.optimize import minimize
    out = {"rows": int(len(R)), "train_months": sorted(set(mon[tr])), "test_months": sorted(set(mon[te]))}
    for name, cols in {"A1 場との相性": [0], "A2 場×コースの相性": [1], "A3 平均着順の差": [2], "全部": [0, 1, 2]}.items():
        def f(bb, m):
            beta = np.zeros(3); beta[cols] = bb
            return nll(beta, m)
        res = minimize(lambda bb: f(bb, tr), np.zeros(len(cols)), method="Nelder-Mead")
        base_te, new_te = f(np.zeros(len(cols)), te), f(res.x, te)
        out[name] = {"beta": [round(float(v), 4) for v in res.x], "base_test": round(base_te, 5), "new_test": round(new_te, 5), "improve": round(base_te - new_te, 5),
                     "base_train": round(f(np.zeros(len(cols)), tr), 5), "new_train": round(float(res.fun), 5)}
        log(name, json.dumps(out[name], ensure_ascii=False))
    # 3) 予想の確率と実際の1着率のずれを、相性の大きさ別に（確かめる期間）
    p = np.exp(lp)
    for j, name in enumerate(["A1", "A2", "A3"]):
        q = np.quantile(X[te, j], [0, 0.1, 0.3, 0.7, 0.9, 1])
        tab = []
        for lo, hi in zip(q[:-1], q[1:]):
            m = te & (X[:, j] >= lo) & (X[:, j] <= hi)
            tab.append({"n": int(m.sum()), "pred": round(float(p[m].mean()), 4), "actual": round(float(y[m].mean()), 4)})
        out[f"{name}_buckets"] = tab
        log(name, "相性の小さい順（下位10%・〜30%・中間・〜90%・上位10%）:", " / ".join(f"予想{t['pred']*100:.1f}% 実際{t['actual']*100:.1f}%" for t in tab))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
