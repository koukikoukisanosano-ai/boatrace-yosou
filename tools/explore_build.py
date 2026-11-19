"""「普通は思いつかない買い方」を探すための表を作る（1回作って保存し、探索はこの表だけで行う）。

公平な検証の月ごとの予想（その月より前だけで学習した本番の予想＝展開＋機械学習）と、
確定オッズ（2連単・2連複・3連複・拡連複・3連単）、実際の払い戻し、レースの条件（場・R・グレード・風・波・
進入・選手の級別・勝率・展示タイムなど）を1レース1行にまとめる。

  python tools/explore_build.py <月ごとの予想のフォルダ> data/explore.npz
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
KEYS = {"exacta": ("e", [f"{a}-{b}" for a, b in itertools.permutations(B, 2)]),
        "quinella": ("q", [f"{a}={b}" for a, b in itertools.combinations(B, 2)]),
        "trio": ("f", [f"{a}={b}={c}" for a, b, c in itertools.combinations(B, 3)]),
        "quinella_place": ("k", [f"{a}={b}" for a, b in itertools.combinations(B, 2)]),
        "trifecta": ("t", [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(B, 3)]),
        "win": ("w", [str(a) for a in B])}
CLASS = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def load_day(d):
    po = ROOT / "data" / "odds" / d[:4] / f"{d}.json.gz"
    pa = ROOT / "data" / "openapi" / d[:4] / f"{d}.json.gz"
    od = json.loads(gzip.decompress(po.read_bytes())) if po.exists() else {}
    st = ((json.loads(gzip.decompress(pa.read_bytes())).get("programs") or {}).get("stadiums") or {}) if pa.exists() else {}
    return od, st


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cols = {k: [] for k in ["id", "month", "jcd", "rno", "grade", "day", "hour", "wind", "wave", "weather", "dist", "nA1", "c1", "win1", "loc1", "mot1", "ex_rank1", "st1", "course1"]}
    P = {bt: [] for bt in KEYS}; O = {bt: [] for bt in KEYS}; Y = {bt: [] for bt in KEYS}
    cache = {}
    for f in sorted(src.rglob("fair_*.pkl")):
        b = pickle.load(open(f, "rb"))
        if b["month"] < "2024-11":
            continue
        bp = bet_probs(geo([b[k].astype(float) for k in MIX], list(MIX.values())))
        for i, rid in enumerate(b["ids"]):
            d, j, r = str(rid).split("-")
            if d not in cache:
                cache.clear(); cache[d] = load_day(d)
            od, st = cache[d]
            rc = ((st.get(j) or {}).get("races") or {}).get(r) or {}
            res = rc.get("result") or {}
            pays = res.get("payouts") or {}
            if not pays:
                continue
            o_all = od.get(f"{j}-{r}") or {}
            racers = rc.get("racers") or {}
            pv = rc.get("preview") or {}
            pvr = pv.get("racers") or {}
            r1 = racers.get("1") or {}
            ext = [(pvr.get(str(w)) or {}).get("exhibition_time") for w in B]
            ex1 = None
            if all(isinstance(x, (int, float)) for x in ext):
                ex1 = 1 + sum(x < ext[0] for x in ext)
            ca = rc.get("closed_at") or ""
            cols["id"].append(str(rid)); cols["month"].append(b["month"]); cols["jcd"].append(int(j)); cols["rno"].append(int(r))
            cols["grade"].append(rc.get("grade_number") or 0); cols["day"].append(rc.get("day_number") or 0)
            cols["hour"].append(int(ca[11:13]) if len(ca) >= 13 else -1)
            cols["wind"].append(pv.get("wind_speed") if isinstance(pv.get("wind_speed"), (int, float)) else np.nan)
            cols["wave"].append(pv.get("wave_height") if isinstance(pv.get("wave_height"), (int, float)) else np.nan)
            cols["weather"].append(pv.get("weather_number") or 0)
            cols["dist"].append(rc.get("distance") or 0)
            cols["nA1"].append(sum(1 for x in racers.values() if x.get("rank_number") == 1))
            cols["c1"].append(r1.get("rank_number") or 0)   # 級別 1=A1 2=A2 3=B1 4=B2
            cols["win1"].append(r1.get("national_win_rate") or np.nan)
            cols["loc1"].append(r1.get("local_win_rate") or np.nan)
            cols["mot1"].append(r1.get("motor_top_2_percent") or np.nan)
            cols["ex_rank1"].append(ex1 or 0)
            cols["st1"].append(r1.get("average_start_timing") or np.nan)
            cols["course1"].append((pvr.get("1") or {}).get("course_number") or 0)
            for bt, (ok, keys) in KEYS.items():
                pr = bp.get(bt)
                P[bt].append(pr[i].astype(np.float32) if pr is not None else np.full(len(keys), np.nan, np.float32))
                o = o_all.get(ok)
                if o and bt in ("quinella_place",):
                    o = [x[0] if isinstance(x, list) else x for x in o]
                if o and bt == "win":
                    o = o
                O[bt].append(np.array([x if isinstance(x, (int, float)) and x else np.nan for x in o], np.float32) if o and len(o) == len(keys) else np.full(len(keys), np.nan, np.float32))
                pm = {x["combination"]: (x.get("amount") or 0) / 100 for x in (pays.get(bt) or []) if x.get("combination")}
                Y[bt].append(np.array([pm.get(k, 0.0) for k in keys], np.float32))
    arr = {k: np.array(v) for k, v in cols.items()}
    for bt in KEYS:
        arr[f"p_{bt}"] = np.stack(P[bt]); arr[f"o_{bt}"] = np.stack(O[bt]); arr[f"y_{bt}"] = np.stack(Y[bt])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arr)
    print("RESULT races", len(arr["id"]), "months", sorted(set(arr["month"]))[0], "〜", sorted(set(arr["month"]))[-1],
          {bt: int(np.isfinite(arr[f"o_{bt}"]).any(1).sum()) for bt in KEYS}, f"{out.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
