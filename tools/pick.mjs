// 締切1分前の買い目を、予想ページと同じ計算で決める（自動取得から呼ぶ → 通知に使う）
//   node tools/pick.mjs <Open API の今日のJSON> <docs/live/YYYYMMDD.json> <キー,キー,…（"場-レース"）>
// 出力（1行1レースのJSON）: {key, venue, rno, closedAt, state: buy|pass|none, picks: [{combo, p, odds, ev}]}
import fs from 'node:fs';
import path from 'node:path';
import { normalizeDay, prepareModel, predictRace, Snapshot, previewState, evPicks } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const [dayPath, livePath, keysArg] = process.argv.slice(2);
const keys = new Set((keysArg || '').split(',').filter(Boolean));
const model = prepareModel(JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'model', 'model.json'), 'utf8')));
const live = JSON.parse(fs.readFileSync(livePath, 'utf8'));
const races = normalizeDay(JSON.parse(fs.readFileSync(dayPath, 'utf8'))).filter((r) => keys.has(`${r.jcd}-${r.rno}`));
const ymd = races.length ? races[0].date.replace(/-/g, '') : '';
const stateFile = [path.join(ROOT, 'docs', 'data', 'state', `${ymd}.json`), path.join(ROOT, 'docs', 'data', 'state', 'latest.json')].find((f) => fs.existsSync(f));
const st = stateFile ? JSON.parse(fs.readFileSync(stateFile, 'utf8')) : null;
const snap = new Snapshot(st ? (st.snapshot || st) : null);
const rule = model.bet ? { ...model.bet.rule, alpha: 1 } : null;

for (const r of races) {
  const key = `${r.jcd}-${r.rno}`;
  // 自動取得した直前情報とオッズを重ねる（ページの mergeLive と同じ）
  const o = live.odds && live.odds[key];
  if (o && (o.t || o.e)) r.odds = { t: o.t || null, e: o.e || null };
  const L = live.races && live.races[key];
  if (L && r.preview !== 'full') {
    for (const b of r.boats) {
      const x = L.boats && L.boats[b.waku];
      if (!x) continue;
      b.exCourse = x.course ?? null; b.exST = x.st ?? null; b.exTime = x.time ?? null; b.tilt = x.tilt ?? null;
      b.exWeight = x.weight ?? null; b.weightAdj = x.adj ?? null; b.propNew = x.prop ? 1 : 0; b.parts = x.parts || [];
    }
    for (const k of ['wind', 'windDir', 'wave', 'weather', 'airTemp', 'waterTemp']) if (L[k] !== undefined && L[k] !== null) r[k] = L[k];
    r.preview = previewState(r);
  }
  let out = { key, venue: r.venue, rno: r.rno, closedAt: r.closedAt, state: 'none', picks: [] };
  try {
    const pr = r.preview !== 'none' ? predictRace(r, model, snap) : null;
    const res = pr && rule && r.odds && r.odds.e ? evPicks(pr, r, rule, 'exacta') : null;
    if (res) out = { ...out, state: res.picks.length ? 'buy' : 'pass', picks: res.picks.map((x) => ({ combo: x.combo, p: +x.p.toFixed(4), odds: x.odds, ev: +x.ev.toFixed(3) })) };
  } catch (e) { out.error = String(e); }
  console.log(JSON.stringify(out));
}
