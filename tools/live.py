"""直前情報の自動取得（GitHub Actions で数時間動かし続ける）

締切が近いレースの「直前情報」ページ（展示タイム・スタート展示・チルト・部品交換・気象）を
BOAT RACE 公式サイトから取り、docs/live/YYYYMMDD.json に書いて公開する。
公開データ（Boatrace Open API）は約10分おきの更新で間に合わないことがあるため、その穴を埋める。

  python tools/live.py [--max-minutes 330] [--once] [--no-push]

相手サーバーへの配慮: 1回ずつ順番に、間隔を空けて取得する。対象は締切20分前〜締切の
レースで、取れたレースはもう取りに行かない。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unicodedata
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path  # noqa: F811

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import odds_parse  # noqa: E402

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # pragma: no cover
    pass

JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "docs" / "live"
OFFICIAL = "https://www.boatrace.jp/owpc/pc/race"
OPENAPI = "https://boatraceopenapi.github.io/api/v1"
UA = "Mozilla/5.0 (compatible; boatrace-yosou/1.0; +https://github.com/koukikoukisanosano-ai/boatrace-yosou)"
INTERVAL = 1.0          # 公式サイトへのリクエストを始める間隔（秒）。1回の応答に数秒かかるので3本まで並べる
WORKERS = 3             # 同時に取りに行く数
WINDOW_BEFORE = 20      # 締切の何分前から取りに行くか
RETRY_SEC = 70          # まだ出ていなかったレースを次に見に行くまでの秒数（締切10分前を切ったら40秒）
ODDS_FROM, ODDS_TO = 7.0, 2.0   # 締切の何分前のオッズ（3連単）を取るか（この間に1回）


def log(*a):
    print(datetime.now(JST).strftime("%H:%M:%S"), *a, flush=True)


def norm(s: str | None) -> str:
    return unicodedata.normalize("NFKC", s or "").replace("\xa0", " ").strip()


def fnum(s: str | None):
    m = re.search(r"-?(?:\d+(?:\.\d+)?|\.\d+)", norm(s))
    return float(m.group()) if m else None


# ---------------------------------------------------------------- 直前情報ページの読み取り
def parse_beforeinfo(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    boats: dict[str, dict] = {}
    for tb in soup.select("div.table1 table.is-w748 tbody"):
        trs = tb.find_all("tr", recursive=False)
        if not trs:
            continue
        tds = trs[0].find_all("td", recursive=False)
        if len(tds) < 8:
            continue
        w = fnum(tds[0].get_text())
        if w is None:
            continue
        parts = []
        for li in tds[7].select("li"):
            t = norm(li.get_text())
            m = re.match(r"(.+?)(?:×(\d+))?$", t)
            if m:
                parts.append(m.group(1) + (m.group(2) or ""))
        adj = None
        if len(trs) >= 3:
            t3 = trs[2].find_all("td", recursive=False)
            if t3:
                adj = fnum(t3[0].get_text())
        prop = norm(tds[6].get_text())
        boats[str(int(w))] = {
            "weight": fnum(tds[3].get_text()), "time": fnum(tds[4].get_text()), "tilt": fnum(tds[5].get_text()),
            "prop": 1 if prop else 0, "parts": parts, "adj": adj,
        }
    # スタート展示: 上から順に進入コース
    for course, box in enumerate(soup.select("div.table1_boatImage1"), start=1):
        n = box.select_one(".table1_boatImage1Number")
        t = box.select_one(".table1_boatImage1Time")
        w = fnum(n.get_text()) if n else None
        if w is None:
            continue
        txt = norm(t.get_text()) if t else ""
        b = boats.setdefault(str(int(w)), {})
        b["course"] = course
        v = fnum(txt.lstrip("FL"))
        if txt.startswith("F") and v is not None:
            b["st"] = -abs(v)
        elif txt.startswith("L"):
            b["st"] = None
        else:
            b["st"] = v
    out: dict = {"boats": boats}
    for unit in soup.select("div.weather1_bodyUnit"):
        cls = " ".join(unit.get("class", []))
        title = norm(unit.select_one(".weather1_bodyUnitLabelTitle").get_text()) if unit.select_one(".weather1_bodyUnitLabelTitle") else ""
        data = norm(unit.select_one(".weather1_bodyUnitLabelData").get_text()) if unit.select_one(".weather1_bodyUnitLabelData") else ""
        img = unit.select_one("p.weather1_bodyUnitImage")
        icls = " ".join(img.get("class", [])) if img is not None else ""
        if "is-weather" in cls and "is-wind" not in cls:
            m = re.search(r"is-weather(\d+)", icls)
            out["weather"] = int(m.group(1)) if m else None
        elif "is-windDirection" in cls:
            m = re.search(r"is-wind(\d+)", icls)
            out["windDir"] = int(m.group(1)) if m else None
        elif "風速" in title:
            out["wind"] = fnum(data)
        elif "気温" in title:
            out["airTemp"] = fnum(data)
        elif "水温" in title:
            out["waterTemp"] = fnum(data)
        elif "波高" in title:
            out["wave"] = fnum(data)
    return out


def complete(info: dict) -> bool:
    """展示が終わって、表示がそろっているか（欠場艇は除く）"""
    bs = [b for b in info.get("boats", {}).values()]
    have = [b for b in bs if b.get("time") is not None and b.get("course") is not None]
    return len(have) >= 3 and len(have) >= len([b for b in bs if b.get("weight") is not None]) - 1 and \
        all(b.get("course") is not None for b in have)


# ---------------------------------------------------------------- 本体
class Live:
    def __init__(self, push: bool):
        self.push = push
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
        self.last_req = 0.0
        self.day = ""
        self.store: dict = {}
        self.schedule: list[dict] = []
        self.sched_at = 0.0
        self.tried: dict[str, float] = {}
        self.dirty = False
        self.unpushed = False
        self.need_deploy = False
        self.last_push = 0.0
        self.stats = {"requests": 0, "captured": 0, "errors": 0}
        self.lock = threading.Lock()
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=WORKERS)

    def path(self) -> Path:
        return LIVE_DIR / f"{self.day.replace('-', '')}.json"

    def load(self, day: str):
        self.day = day
        p = self.path()
        self.store = json.loads(p.read_text()) if p.exists() else {"date": day, "updated": None, "races": {}}
        self.store.setdefault("odds", {})
        self.tried = {}
        self.odds_tried = {}

    def session(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = self.local.s = requests.Session()
            s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
        return s

    def official(self, page: str, jcd: int, rno: int) -> str | None:
        with self.lock:  # 全体で間隔を空ける
            wait = INTERVAL - (time.time() - self.last_req)
            if wait > 0:
                time.sleep(wait)
            self.last_req = time.time()
            self.stats["requests"] += 1
        try:
            r = self.session().get(f"{OFFICIAL}/{page}", params={"rno": rno, "jcd": f"{jcd:02d}", "hd": self.day.replace("-", "")}, timeout=40)
            if r.status_code != 200:
                self.stats["errors"] += 1
                log(f"  {page} {jcd}-{rno}: HTTP {r.status_code}")
                return None
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException as e:
            self.stats["errors"] += 1
            log(f"  {page} {jcd}-{rno}: {e}")
            return None

    def refresh_schedule(self):
        """今日のレースと締切時刻、公開データ側の直前情報の有無（Open API）"""
        ymd = self.day
        try:
            r = self.s.get(f"{OPENAPI}/{ymd[:4]}/{ymd.replace('-', '')}.json", params={"t": int(time.time())}, timeout=60)
            if r.status_code == 404:
                r = self.s.get(f"{OPENAPI}/today.json", params={"t": int(time.time())}, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception as e:  # noqa: BLE001
            log("公開データの取得に失敗:", e)
            self.sched_at = time.time() - 420  # 3分後にもう一度
            return
        sched = []
        for jcd, st in (j.get("programs", {}).get("stadiums") or {}).items():
            for rno, rc in (st.get("races") or {}).items():
                if (rc.get("date") or "") != ymd or not rc.get("closed_at"):
                    continue
                pv = (rc.get("preview") or {}).get("racers") or {}
                has = sum(1 for x in pv.values() if x.get("exhibition_time") is not None and x.get("course_number") is not None)
                dl = datetime.fromisoformat(rc["closed_at"]).replace(tzinfo=JST)
                sched.append({"jcd": int(jcd), "rno": int(rno), "deadline": dl, "openapi_has": has >= len(pv) and has > 0})
        sched.sort(key=lambda x: x["deadline"])
        self.schedule = sched
        self.sched_at = time.time()
        log(f"レース予定 {len(sched)} 件（公開データに直前情報あり {sum(x['openapi_has'] for x in sched)} 件）")

    def targets(self, now: datetime) -> list[dict]:
        out = []
        for x in self.schedule:
            key = f"{x['jcd']}-{x['rno']}"
            mins = (x["deadline"] - now).total_seconds() / 60
            if mins > WINDOW_BEFORE or mins < -2:
                continue
            if key in self.store["races"] or x["openapi_has"]:
                continue
            retry = RETRY_SEC if mins > 10 else 40
            if time.time() - self.tried.get(key, 0) < retry:
                continue
            out.append(x)
        return out

    def odds_targets(self, now: datetime) -> list[dict]:
        """締切7〜2分前で、まだオッズを取っていないレース"""
        out = []
        for x in self.schedule:
            key = f"{x['jcd']}-{x['rno']}"
            mins = (x["deadline"] - now).total_seconds() / 60
            if not (ODDS_TO <= mins <= ODDS_FROM) or key in self.store["odds"]:
                continue
            if time.time() - self.odds_tried.get(key, 0) < 50:
                continue
            out.append(x)
        return out

    def fetch_odds(self, x: dict):
        html = self.official("odds3t", x["jcd"], x["rno"])
        od = odds_parse.parse("odds3t", html) if html else None
        if od and od.get("t"):
            # 2連単（と2連複）も取る。取れなくても3連単だけで使う
            h2 = self.official("odds2tf", x["jcd"], x["rno"])
            if h2:
                try:
                    o2 = odds_parse.parse("odds2tf", h2)
                    od["e"], od["q"] = o2.get("e"), o2.get("q")
                except Exception as e:  # noqa: BLE001
                    log("2連単オッズの読み取り失敗", e)
        return x, od

    def capture_odds(self, xs: list[dict]):
        for x in xs:
            self.odds_tried[f"{x['jcd']}-{x['rno']}"] = time.time()
        for x, od in self.pool.map(self.fetch_odds, xs):
            if not od or not od.get("t") or sum(v is not None for v in od["t"]) < 60:
                continue
            key = f"{x['jcd']}-{x['rno']}"
            now = datetime.now(JST)
            rec = {"at": now.isoformat(timespec="seconds"), "site": od.get("at"), "t": od["t"]}
            if od.get("e") and sum(v is not None for v in od["e"]) >= 15:
                rec["e"] = od["e"]
            if od.get("q") and sum(v is not None for v in od["q"]) >= 8:
                rec["q"] = od["q"]
            self.store["odds"][key] = rec
            self.store["updated"] = now.isoformat(timespec="seconds")
            self.dirty = True
            self.stats["odds"] = self.stats.get("odds", 0) + 1
            log(f"オッズ {key} 締切{x['deadline']:%H:%M}（あと{(x['deadline'] - now).total_seconds() / 60:.1f}分）")

    def fetch(self, x: dict):
        html = self.official("beforeinfo", x["jcd"], x["rno"])
        return x, (parse_beforeinfo(html) if html else None)

    def capture_many(self, xs: list[dict]):
        for x in xs:
            self.tried[f"{x['jcd']}-{x['rno']}"] = time.time()
        for x, info in self.pool.map(self.fetch, xs):
            if info is not None:
                self.store_info(x, info)

    def store_info(self, x: dict, info: dict):
        key = f"{x['jcd']}-{x['rno']}"
        if not complete(info):
            return
        info["at"] = datetime.now(JST).isoformat(timespec="seconds")
        self.store["races"][key] = info
        self.store["updated"] = info["at"]
        self.dirty = True
        self.stats["captured"] += 1
        mins = (x["deadline"] - datetime.now(JST)).total_seconds() / 60
        log(f"取得 {key} 締切{x['deadline']:%H:%M}（あと{mins:.1f}分）")

    def git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    def publish(self):
        """書き出して保存（push）し、ページの公開を依頼する。push できなかった分は次の回にまとめて送る"""
        if self.dirty:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            self.path().write_text(json.dumps(self.store, ensure_ascii=False, separators=(",", ":")))
            self.dirty = False
            if self.push:
                rel = str(self.path().relative_to(ROOT))
                self.git("add", rel)
                if self.git("diff", "--cached", "--quiet").returncode != 0:
                    self.git("commit", "-q", "-m", f"直前情報 {datetime.now(JST):%H:%M}")
                    self.unpushed = True
        if not self.push or not getattr(self, "unpushed", False):
            return
        for i in range(4):
            pull = self.git("pull", "-q", "--rebase", "-X", "theirs")
            if pull.returncode != 0:
                self.git("rebase", "--abort")
            r = self.git("push", "-q")
            if r.returncode == 0:
                self.unpushed = False
                break
            log("保存に失敗（やり直します）:", (r.stderr or pull.stderr).strip().splitlines()[-1:] )
            time.sleep(3 * (i + 1))
        if self.unpushed:
            return  # 次のループでもう一度
        self.last_push = time.time()
        self.need_deploy = True

    def deploy(self):
        """GitHub Pages に反映（別のワークフローで公開）。失敗したら次のループでもう一度"""
        if not self.need_deploy or not self.push:
            return
        r = subprocess.run(["gh", "workflow", "run", "pages.yml", "--ref", "main"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            log("公開の依頼に失敗（次の回にもう一度）:", r.stderr.strip()[:200])
        else:
            self.need_deploy = False
            log("保存して公開を依頼しました")

    def run(self, max_minutes: float, once: bool):
        started = time.time()
        while True:
            now = datetime.now(JST)
            day = now.strftime("%Y-%m-%d")
            if day != self.day:
                self.load(day)
                self.refresh_schedule()
            elif time.time() - self.sched_at > 600:
                self.refresh_schedule()
            # 今日の最後のレースが終わったら終了
            if self.schedule and now > self.schedule[-1]["deadline"] + timedelta(minutes=5):
                log("今日のレースは終わりました")
                break
            self.capture_odds(self.odds_targets(now)[:6])   # 締切が近いので先に
            self.capture_many(self.targets(now)[:12])
            self.publish()
            self.deploy()
            if once:
                break
            if time.time() - started > max_minutes * 60:
                log("時間になったので終了（次の実行に引き継ぎ）")
                break
            time.sleep(20)
        self.publish()
        self.deploy()
        log("集計:", json.dumps(self.stats, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=330)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--parse", help="保存したHTMLを読み取って表示（確認用）")
    a = ap.parse_args()
    if a.parse:
        print(json.dumps(parse_beforeinfo(Path(a.parse).read_text(encoding="utf-8")), ensure_ascii=False, indent=1))
        return 0
    Live(push=not a.no_push).run(a.max_minutes, a.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
