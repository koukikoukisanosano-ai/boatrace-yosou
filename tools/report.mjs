// 毎日の成績（答え合わせ）を記録する: docs/data/report.json
// その日の朝に用意した近況と、その日より前までで学習したモデルで予想し直すので、ずるの無い成績になる
//   node tools/report.mjs [YYYY-MM-DD ...]   (省略時は昨日)
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { normalizeDay, Snapshot, prepareModel, predictRace, evPicks } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const OUT = path.join(ROOT, 'docs', 'data', 'report.json');
const jst = (offsetDays = 0) => new Date(Date.now() + 9 * 3600e3 + offsetDays * 86400e3).toISOString().slice(0, 10);
const dates = process.argv.slice(2).filter((a) => /^\d{4}-\d{2}-\d{2}$/.test(a));
if (!dates.length) dates.push(jst(-1));

const model = prepareModel(JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'model', 'model.json'), 'utf8')));
const report = fs.existsSync(OUT) ? JSON.parse(fs.readFileSync(OUT, 'utf8')) : { days: [] };
const researchFile = path.join(ROOT, 'docs', 'data', 'research.json');
const recs = fs.existsSync(researchFile) ? (JSON.parse(fs.readFileSync(researchFile, 'utf8')).recommend || []) : [];
// 2連単の買い方（学習で選んだ決め方: model.bet）。予想はオッズを混ぜない
const EV_RULES = model.bet ? [{ kind: 'exacta', title: '2連単 期待値の買い目（メイン）', rule: { ...model.bet.rule, alpha: 1 } }] : [];
const LIST = { trifecta: 'trifecta', trio: 'trio', exacta: 'exacta', quinella: 'quinella', quinella_place: 'wide', win: 'win', place: 'place' };

for (const ymd of dates) {
  const compact = ymd.replace(/-/g, '');
  if (model.trained_until && model.trained_until >= ymd) { console.error(`${ymd}: モデルがこの日を学習済みなので記録しません`); continue; }
  const stateFile = path.join(ROOT, 'docs', 'data', 'state', `${compact}.json`);
  const dayFile = path.join(ROOT, 'data', 'openapi', ymd.slice(0, 4), `${compact}.json.gz`);
  if (!fs.existsSync(stateFile) || !fs.existsSync(dayFile)) { console.error(`${ymd}: 近況か結果のデータがありません`); continue; }
  const st = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  const snap = new Snapshot(st.snapshot || st);
  const races = normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(dayFile)).toString('utf8')));
  // その日に締切前に取ったオッズ（ページと同じように予想に混ぜる）
  const liveFile = path.join(ROOT, 'docs', 'live', `${compact}.json`);
  const live = fs.existsSync(liveFile) ? JSON.parse(fs.readFileSync(liveFile, 'utf8')) : null;
  if (live && live.odds) for (const r of races) { const o = live.odds[`${r.jcd}-${r.rno}`]; if (o && o.t) r.odds = { t: o.t, e: o.e }; }
  const s = { date: ymd, model: model.version, races: 0, top1: 0, tri1: 0, tri1Ret: 0, tri5: 0, tri5Ret: 0, ex1: 0, ex1Ret: 0, withOdds: 0, rec: {} };
  for (const r of races) {
    if (!r.hasResult || r.preview === 'none') continue;
    const tp = r.payouts.trifecta || [];
    if (!tp.length) continue;
    const pr = predictRace(r, model, snap);
    if (!pr) continue;
    s.races++;
    const winner = r.boats.find((b) => b.place === 1);
    const best = pr.probs.win.indexOf(Math.max(...pr.probs.win));
    if (winner && pr.boats[best].waku === winner.waku) s.top1++;
    const pay = (list, combo) => { const x = list.find((p) => p.combo === combo); return x ? x.amount : 0; };
    const t1 = pay(tp, pr.trifecta[0].combo);
    if (t1) { s.tri1++; s.tri1Ret += t1; }
    let hit5 = false;
    for (const t of pr.trifecta.slice(0, 5)) { const a = pay(tp, t.combo); if (a) { hit5 = true; s.tri5Ret += a; } }
    if (hit5) s.tri5++;
    const e1 = pay(r.payouts.exacta || [], pr.exacta[0].combo);
    if (e1) { s.ex1++; s.ex1Ret += e1; }
    if (pr.withOdds) s.withOdds++;
    // 期待値の買い方の成績（締切前に取ったオッズで選び、払戻で答え合わせ）
    const refundsN = (r.refunds || []).map(Number);
    for (const er of EV_RULES) {
      const res = evPicks(pr, r, er.rule, er.kind);
      if (!res) continue;
      const x = s.rec[er.title] || (s.rec[er.title] = { bets: 0, hits: 0, ret: 0, races: 0 });
      x.races++;
      for (const t of res.picks) {
        x.bets++;
        if (t.combo.split('-').map(Number).some((b) => refundsN.includes(b))) { x.ret += 100; continue; }
        const a = pay(r.payouts[er.kind] || [], t.combo);
        if (a) { x.hits++; x.ret += a; }
      }
    }
    // おすすめの買い方の成績
    for (const rc of (model.bet ? [] : recs)) {
      const list = pr[LIST[rc.bet]] || [];
      if (!list.length || list[0].p < (rc.rule.conf || 0)) continue;
      const x = s.rec[rc.title] || (s.rec[rc.title] = { bets: 0, hits: 0, ret: 0 });
      const paid = r.payouts[rc.bet] || [];
      const refunds = (r.refunds || []).map(Number);
      for (const t of list.slice(0, rc.rule.k)) {
        x.bets++;
        if (t.combo.split(/[-=]/).map(Number).some((b) => refunds.includes(b))) { x.ret += 100; continue; }
        const a = pay(paid, t.combo);
        if (a) { x.hits++; x.ret += a; }
      }
    }
  }
  if (!s.races) { console.error(`${ymd}: 対象レースなし`); continue; }
  report.days = report.days.filter((d) => d.date !== ymd).concat([s]).sort((a, b) => (a.date < b.date ? -1 : 1)).slice(-120);
  console.log(`${ymd}: ${s.races}レース 1着的中 ${(100 * s.top1 / s.races).toFixed(1)}% 3連単上位5点 的中 ${(100 * s.tri5 / s.races).toFixed(1)}% 回収 ${(s.tri5Ret / (s.races * 500) * 100).toFixed(0)}%`);
}
report.updated = new Date().toISOString();
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(report));
