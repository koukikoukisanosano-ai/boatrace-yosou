"""確かめる: 予想の確率1番の目と、オッズ1番人気の目が違うレースで、予想の1番を買ったらどうなるか。
公平な検証の月ごとの予想（その月より前だけで学習）と、確定オッズ・実際の払い戻し（2026年1〜10月）。
  python tools/disagree_test.py <月ごとの予想のフォルダ>
"""
import gzip, itertools, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eval import bet_probs  # noqa: E402
from research_exacta import geo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIX = {"tenkai_big": 0.5, "gbm": 0.5}
B = range(1, 7)
TYPES = {"2連単": ("exacta", "e", [f"{a}-{b}" for a, b in itertools.permutations(B, 2)]),
         "2連複": ("quinella", "q", [f"{a}={b}" for a, b in itertools.combinations(B, 2)]),
         "3連複": ("trio", "f", [f"{a}={b}={c}" for a, b, c in itertools.combinations(B, 3)]),
         "3連単": ("trifecta", "t", [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(B, 3)])}
cache = {}
rows = {k: [] for k in TYPES}
for f in sorted(Path(sys.argv[1]).rglob("fair_*.pkl")):
    b = pickle.load(open(f, "rb"))
    if b["month"] < "2026-01":
        continue
    bp = bet_probs(geo([b[k].astype(float) for k in MIX], list(MIX.values())))
    for i, rid in enumerate(b["ids"]):
        d, j, r = str(rid).split("-")
        if d not in cache:
            po = ROOT / "data" / "odds" / d[:4] / f"{d}.json.gz"
            pa = ROOT / "data" / "openapi" / d[:4] / f"{d}.json.gz"
            cache.clear()
            cache[d] = (json.loads(gzip.decompress(po.read_bytes())) if po.exists() else {},
                        ((json.loads(gzip.decompress(pa.read_bytes())).get("programs") or {}).get("stadiums") or {}) if pa.exists() else {})
        od, st = cache[d]
        rc = ((st.get(j) or {}).get("races") or {}).get(r) or {}
        pays = (rc.get("result") or {}).get("payouts") or {}
        o_all = od.get(f"{j}-{r}") or {}
        for name, (bt, ok, keys) in TYPES.items():
            o = o_all.get(ok); pp = pays.get(bt) or []
            if not o or not pp:
                continue
            o = np.array([x if x else np.nan for x in o], float)
            if np.isnan(o).all():
                continue
            pm = {x["combination"]: x["amount"] / 100 for x in pp}
            p = bp[bt][i]
            mi = int(np.argmax(p)); ki = int(np.nanargmin(o))
            rows[name].append((mi == ki, p[mi], o[mi], pm.get(keys[mi], 0), o[ki], pm.get(keys[ki], 0)))
for name, R in rows.items():
    A = np.array(R, float)
    agree, p, om, ym, ok_, yk = A.T
    dis = agree == 0
    print(f"RESULT ===== {name}: {len(A)}レース  予想の1番＝人気の1番 {agree.mean() * 100:.0f}%")
    for lab, m in (("一致したレース", agree == 1), ("違ったレース", dis)):
        print(f"RESULT {lab}: {int(m.sum())}レース 予想1番を買う→的中率{(ym[m] > 0).mean() * 100:.1f}%（予想{p[m].mean() * 100:.1f}%） 平均{np.median(om[m]):.1f}倍 回収率{ym[m].mean() * 100:.1f}%"
              + (f" ｜人気1番を買う→的中率{(yk[m] > 0).mean() * 100:.1f}% 回収率{yk[m].mean() * 100:.1f}%" if lab == "違ったレース" else ""))
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 1e9)):
        m = dis & (om >= lo) & (om < hi)
        if m.sum() >= 30:
            print(f"RESULT   違ったレースで予想1番のオッズ{lo}〜{hi if hi < 1e9 else ''}倍: {int(m.sum())}レース 的中率{(ym[m] > 0).mean() * 100:.1f}% 回収率{ym[m].mean() * 100:.1f}%")
