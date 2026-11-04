// 毎日の成績（答え合わせ）を記録する: docs/data/report.json
// その日の朝に用意した近況と、その日より前までで学習したモデルで予想し直すので、ずるの無い成績になる
//   node tools/report.mjs [YYYY-MM-DD ...]   (省略時は昨日)
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { normalizeDay, Snapshot, prepareModel, predictRace } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const OUT = path.join(ROOT, 'docs', 'data', 'report.json');
const jst = (offsetDays = 0) => new Date(Date.now() + 9 * 3600e3 + offsetDays * 86400e3).toISOString().slice(0, 10);
const dates = process.argv.slice(2).filter((a) => /^\d{4}-\d{2}-\d{2}$/.test(a));
if (!dates.length) dates.push(jst(-1));

const model = prepareModel(JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'model', 'model.json'), 'utf8')));
const report = fs.existsSync(OUT) ? JSON.parse(fs.readFileSync(OUT, 'utf8')) : { days: [] };

for (const ymd of dates) {
  const compact = ymd.replace(/-/g, '');
  if (model.trained_until && model.trained_until >= ymd) { console.error(`${ymd}: モデルがこの日を学習済みなので記録しません`); continue; }
  const stateFile = path.join(ROOT, 'docs', 'data', 'state', `${compact}.json`);
  const dayFile = path.join(ROOT, 'data', 'openapi', ymd.slice(0, 4), `${compact}.json.gz`);
  if (!fs.existsSync(stateFile) || !fs.existsSync(dayFile)) { console.error(`${ymd}: 近況か結果のデータがありません`); continue; }
  const st = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  const snap = new Snapshot(st.snapshot || st);
  const races = normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(dayFile)).toString('utf8')));
  const s = { date: ymd, model: model.version, races: 0, top1: 0, tri1: 0, tri1Ret: 0, tri5: 0, tri5Ret: 0, ex1: 0, ex1Ret: 0 };
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
  }
  if (!s.races) { console.error(`${ymd}: 対象レースなし`); continue; }
  report.days = report.days.filter((d) => d.date !== ymd).concat([s]).sort((a, b) => (a.date < b.date ? -1 : 1)).slice(-120);
  console.log(`${ymd}: ${s.races}レース 1着的中 ${(100 * s.top1 / s.races).toFixed(1)}% 3連単上位5点 的中 ${(100 * s.tri5 / s.races).toFixed(1)}% 回収 ${(s.tri5Ret / (s.races * 500) * 100).toFixed(0)}%`);
}
report.updated = new Date().toISOString();
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(report));
