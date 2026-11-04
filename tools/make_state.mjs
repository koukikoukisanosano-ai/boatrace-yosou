// 指定日に走る選手・モーターの近況（前日までの結果）を書き出す
//   node tools/make_state.mjs [YYYY-MM-DD ...]   (省略時は今日)
// 出力: docs/data/state/YYYYMMDD.json と docs/data/state/latest.json
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { normalizeDay, History, dayNumber } from '../docs/js/core.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const OUT = path.join(ROOT, 'docs', 'data', 'state');
const OPENAPI = 'https://boatraceopenapi.github.io/api/v1';
fs.mkdirSync(OUT, { recursive: true });

const jstToday = () => new Date(Date.now() + 9 * 3600e3).toISOString().slice(0, 10);
const targets = process.argv.slice(2).filter((a) => /^\d{4}-\d{2}-\d{2}$/.test(a));
if (!targets.length) targets.push(jstToday());

function readLocal(ymd) {
  const p = path.join(ROOT, 'data', 'openapi', ymd.slice(0, 4), `${ymd.replace(/-/g, '')}.json.gz`);
  return fs.existsSync(p) ? JSON.parse(zlib.gunzipSync(fs.readFileSync(p)).toString('utf8')) : null;
}

async function fetchDay(ymd) {
  const urls = [`${OPENAPI}/${ymd.slice(0, 4)}/${ymd.replace(/-/g, '')}.json`];
  if (ymd === jstToday()) urls.push(`${OPENAPI}/today.json`);
  for (const u of urls) {
    for (let i = 0; i < 3; i++) {
      try {
        const r = await fetch(u, { headers: { 'User-Agent': 'boatrace-yosou/1.0' } });
        if (r.status === 404) break;
        if (r.ok) {
          const j = await r.json();
          const races = normalizeDay(j);
          if (races.length && races[0].date === ymd) return races;
          break;
        }
      } catch (e) { console.error(`取得失敗 ${u}: ${e.message}`); }
      await new Promise((res) => setTimeout(res, 3000 * (i + 1)));
    }
  }
  const local = readLocal(ymd);
  return local ? normalizeDay(local) : [];
}

// 前日までの結果で近況を作る
const hist = new History();
const warm = path.join(ROOT, 'data', 'warmup');
if (fs.existsSync(warm)) {
  for (const fn of fs.readdirSync(warm).filter((x) => x.endsWith('.jsonl.gz')).sort()) {
    for (const ln of zlib.gunzipSync(fs.readFileSync(path.join(warm, fn))).toString('utf8').split('\n')) if (ln) hist.addRace(JSON.parse(ln));
  }
}
const base = path.join(ROOT, 'data', 'openapi');
const files = [];
for (const y of fs.readdirSync(base).sort()) for (const fn of fs.readdirSync(path.join(base, y)).sort()) files.push(path.join(base, y, fn));
const first = targets.slice().sort()[0];
for (const f of files) {
  const d = path.basename(f).slice(0, 8);
  const ymd = `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
  if (ymd >= first) break;
  for (const r of normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(f)).toString('utf8')))) hist.addRace(r);
}

for (const ymd of targets.sort()) {
  const races = await fetchDay(ymd);
  if (!races.length) { console.error(`${ymd}: 出走表がまだありません`); continue; }
  // 対象日の前日までの結果がそろっているか（間の日を足す）
  for (const f of files) {
    const d = path.basename(f).slice(0, 8);
    const day = `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
    if (day >= first && day < ymd && dayNumber(day) > hist.lastDay) {
      for (const r of normalizeDay(JSON.parse(zlib.gunzipSync(fs.readFileSync(f)).toString('utf8')))) hist.addRace(r);
    }
  }
  const snap = hist.snapshotFor(races);
  const out = { date: ymd, generated: new Date().toISOString(), resultsUntil: new Date(hist.lastDay * 86400000).toISOString().slice(0, 10), snapshot: snap };
  const body = JSON.stringify(out);
  fs.writeFileSync(path.join(OUT, `${ymd.replace(/-/g, '')}.json`), body);
  if (ymd >= jstToday()) fs.writeFileSync(path.join(OUT, 'latest.json'), body);
  console.log(`${ymd}: ${races.length} レース / 選手 ${Object.keys(snap.r).length} 人 / 結果は ${out.resultsUntil} まで (${(body.length / 1024).toFixed(0)} KB)`);
}

// 古いファイルを整理（45日より前）
const keep = dayNumber(jstToday()) - 45;
for (const fn of fs.readdirSync(OUT)) {
  const m = fn.match(/^(\d{8})\.json$/);
  if (m && dayNumber(m[1]) < keep) fs.unlinkSync(path.join(OUT, fn));
}
