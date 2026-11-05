// 予想ページ: データを読み込んで、スマホの中で予想を計算して表示する
import { normalizeDay, prepareModel, predictRace, Snapshot, previewState, exactaMap, evPicks, formation, FEATURE_INDEX, RANK_LABEL } from './core.js';

const OPENAPI = 'https://boatraceopenapi.github.io/api/v1';
const REFRESH_MS = 60 * 1000;
const WIND = ['', '北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東', '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西', '無風'];
const WEATHER = { 1: '晴', 2: '曇り', 3: '雨', 4: '雪', 5: '台風', 6: '霧' };
const MARKS = ['◎', '○', '▲', '△'];

const state = {
  date: jstToday(), model: null, snap: null, snapNote: '', races: [], preds: new Map(),
  openApiAt: null, liveAt: null, liveCount: 0, view: 'upcoming', venue: 0, open: new Set(),
  error: '', liveError: false, loading: false, loadedAt: null, report: null, snapExact: true, research: null, ev: null, exa: null, sim: null,
  mode: loadPref('mode2', 'roi'),
  bank: +loadPref('bank', '20000') || 20000,
};
function loadPref(k, d) { try { return localStorage.getItem(k) || d; } catch (e) { return d; } }
function savePref(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 保存できなくても動く */ } }
// 予想の重み: 展示最重視（枠＋展示航走のモデルを強く混ぜる）/ 総合（的中率がいちばん高い混ぜ方）
// 予想の重み: 回収率重視（2連単の買い目を選ぶのと同じ予想）/ 展示最重視（枠＋展示航走のモデルを強く混ぜる）
// 1点の購入額: 資金の1/200（過去の検証で一番大きかった連敗の落ち込み 約170点分に耐える額）を100円単位、上限3,000円
const MAX_STAKE = 3000;
const stake = () => Math.min(MAX_STAKE, Math.max(100, Math.floor(state.bank / 200 / 100) * 100));   // 1レースの購入額
const perPoint = (n) => Math.max(100, Math.floor(stake() / n / 100) * 100);                          // 何点か買うときの1点の額
const buyText = (n) => (n > 1 ? `各${yen(perPoint(n))}×${n}点` : yen(stake()));
const yen = (v) => `${Math.round(v).toLocaleString()}円`;
const exhibitW = () => (state.mode === 'exhibit' ? (state.model && state.model.exhibit_w) || 0.85 : 0);

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function jstToday() { return new Date(Date.now() + 9 * 3600e3).toISOString().slice(0, 10); }
function addDays(ymd, n) { const d = new Date(`${ymd}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); }
const compact = (ymd) => ymd.replace(/-/g, '');
function deadline(r) { return new Date(`${r.date}T${r.closedAt || '23:59'}:00+09:00`); }
function minsLeft(r) { return (deadline(r) - Date.now()) / 60000; }
function hhmm(d) { if (!d) return '--:--'; const x = new Date(d.getTime() + 9 * 3600e3); return x.toISOString().slice(11, 16); }
const pct = (p, digits = 0) => (p * 100).toFixed(digits);

async function getJSON(url, cache = 'no-store') {
  const r = await fetch(url, { cache });
  if (!r.ok) { const e = new Error(`HTTP ${r.status}`); e.status = r.status; throw e; }
  const lm = r.headers.get('last-modified');
  return { json: await r.json(), lastModified: lm ? new Date(lm) : null };
}

// ------------------------------------------------------------------ 読み込み
async function loadModel() {
  if (state.model) return;
  const { json } = await getJSON('model/model.json', 'no-cache');
  state.model = prepareModel(json);
}

async function loadSnapshot(ymd) {
  try {
    const { json } = await getJSON(`data/state/${compact(ymd)}.json`, 'no-cache');
    state.snap = new Snapshot(json.snapshot || json);
    state.snapNote = '';
    state.snapExact = true;
  } catch (e) {
    state.snapExact = false;
    try {
      const { json } = await getJSON('data/state/latest.json', 'no-cache');
      state.snap = new Snapshot(json.snapshot || json);
      state.snapNote = json.date ? `選手の近況は ${json.date} 時点` : '';
    } catch (e2) {
      state.snap = new Snapshot(null);
      state.snapNote = '選手の近況データなし';
    }
  }
}

async function loadDay(ymd) {
  let res;
  try {
    res = await getJSON(`${OPENAPI}/${ymd.slice(0, 4)}/${compact(ymd)}.json?t=${Date.now()}`);
  } catch (e) {
    if (ymd === jstToday() && e.status === 404) res = await getJSON(`${OPENAPI}/today.json?t=${Date.now()}`);
    else throw e;
  }
  state.openApiAt = res.lastModified;
  return normalizeDay(res.json);
}

async function loadLive(ymd) {
  try {
    const { json } = await getJSON(`live/${compact(ymd)}.json?t=${Date.now()}`);
    state.liveError = false;
    return json;
  } catch (e) {
    state.liveError = e.status !== 404;
    return null;
  }
}

// 自前で取った直前情報（公式サイトから数分おき）で、公開データにまだ無い分を埋める
function mergeLive(races, live) {
  state.liveAt = live && live.updated ? new Date(live.updated) : null;
  state.liveCount = 0;
  if (!live) return;
  for (const r of races) {
    const o = live.odds && live.odds[`${r.jcd}-${r.rno}`];
    if (o && (o.t || o.e)) r.odds = { t: o.t || null, e: o.e || null, at: o.at ? new Date(o.at) : null, site: o.site || '', final: !!o.final };
    const f = fastCache.get(`${r.jcd}-${r.rno}`);
    if (f && (!r.odds || !r.odds.at || f.at > r.odds.at)) r.odds = f;
  }
  if (!live.races) return;
  for (const r of races) {
    const L = live.races[`${r.jcd}-${r.rno}`];
    if (!L || r.preview === 'full') continue;
    for (const b of r.boats) {
      const x = L.boats && L.boats[b.waku];
      if (!x) continue;
      b.exCourse = x.course ?? null; b.exST = x.st ?? null; b.exTime = x.time ?? null; b.tilt = x.tilt ?? null;
      b.exWeight = x.weight ?? null; b.weightAdj = x.adj ?? null; b.propNew = x.prop ? 1 : 0; b.parts = x.parts || [];
    }
    for (const k of ['wind', 'windDir', 'wave', 'weather', 'airTemp', 'waterTemp']) if (L[k] !== undefined && L[k] !== null) r[k] = L[k];
    const before = r.preview;
    r.preview = previewState(r);
    if (r.preview !== before) { r.previewFrom = 'live'; r.previewAt = L.at ? new Date(L.at) : null; state.liveCount++; }
  }
}

async function refresh(full = false) {
  if (state.loading) return;
  state.loading = true;
  renderStatus();
  try {
    await loadModel();
    if (full || !state.snap) { await loadSnapshot(state.date); predKey.clear(); }
    if (full || !state.report) { try { state.report = (await getJSON('data/report.json', 'no-cache')).json; } catch (e) { state.report = null; } }
    if (!state.sim) { try { state.sim = (await getJSON('data/sim.json', 'no-cache')).json; } catch (e) { state.sim = null; } }
    if (!state.exa) { try { state.exa = (await getJSON('data/research_exacta.json', 'no-cache')).json; } catch (e) { state.exa = null; } }
    const [races, live] = await Promise.all([loadDay(state.date), loadLive(state.date)]);
    mergeLive(races, live);
    state.races = races;
    await predictAll();
    state.error = '';
    state.loadedAt = new Date();
  } catch (e) {
    console.error(e);
    state.error = e.status === 404 ? 'この日のデータはまだありません。' : '公開データを読み込めませんでした。1分ごとに自動で読み込み直します。';
  } finally {
    state.loading = false;
    render();
  }
}

// 予想の計算は重いので、入力（直前情報・オッズ・重み）が変わったレースだけ計算し直す
const predKey = new Map();
let predRun = 0;
async function predictAll() {
  const run = ++predRun;
  const keep = new Map();
  // 締切の近いレースから計算し、少しずつ画面を止めずに進める
  const order = state.races.slice().sort((a, b) => (a.hasResult - b.hasResult) || (deadline(a) - deadline(b)));
  let t = performance.now(); let shown = t;
  for (const r of order) {
    const key = JSON.stringify([state.mode, r.preview, r.odds ? r.odds.site || String(r.odds.at) : '', r.boats.map((b) => [b.exCourse, b.exST, b.exTime, b.tilt, b.weightAdj, b.propNew, (b.parts || []).length, b.place ?? null]), r.wind, r.wave, r.windDir]);
    if (predKey.get(r.id) === key && state.preds.has(r.id)) { keep.set(r.id, state.preds.get(r.id)); continue; }
    try { const p = predictRace(r, state.model, state.snap, { exhibitW: exhibitW() }); if (p) { keep.set(r.id, p); predKey.set(r.id, key); } } catch (e) { console.error(r.id, e); }
    if (performance.now() - t > 40) {
      // 計算の途中でも、できた分から表示する（締切の近いレースが先）
      if (performance.now() - shown > 700 && state.races.length) {
        state.preds = new Map([...state.preds, ...keep]);
        if (!state.loading || state.preds.size) render();
        shown = performance.now();
      }
      await new Promise((res) => setTimeout(res, 0));
      if (run !== predRun) return false;
      t = performance.now();
    }
  }
  if (run !== predRun) return false;
  state.preds = keep;
  return true;
}

// ------------------------------------------------------------------ 表示
function chip(w, cls = '') { return `<span class="chip w${w} ${cls}">${w}</span>`; }
function comboHtml(combo) {
  const parts = combo.split(/([-=])/);
  return `<span class="combo">${parts.map((x) => (x === '-' || x === '=' ? `<span class="sep">${x === '-' ? '–' : '='}</span>` : chip(+x))).join('')}</span>`;
}

function renderStatus() {
  const el = $('#status');
  const bits = [];
  if (state.loading) bits.push('<span><span class="spin"></span> 更新中</span>');
  if (state.openApiAt) bits.push(`<span>公開データ ${hhmm(state.openApiAt)}</span>`);
  if (state.liveAt) bits.push(`<span>直前情報の自動取得 ${hhmm(state.liveAt)}</span>`);
  else if (state.date === jstToday()) bits.push(`<span${state.liveError ? ' class="bad"' : ''}>直前情報の自動取得 ${state.liveError ? '読み込み失敗' : '未開始'}</span>`);
  if (state.snapNote) bits.push(`<span>${esc(state.snapNote)}</span>`);
  el.innerHTML = bits.join('') || '&nbsp;';
  const d = state.date;
  $('#dateLabel').textContent = `${+d.slice(5, 7)}/${+d.slice(8, 10)}${d === jstToday() ? ' 今日' : ''}`;
}

// 締切前: これから締切のレースと、締切から30分以内のレース（結果が出るまで見やすいように残す）
const KEEP_MIN = 30;
function filtered() {
  let rs = state.races.filter((r) => state.venue === 0 || r.jcd === state.venue);
  if (state.view === 'upcoming') rs = rs.filter((r) => minsLeft(r) > -KEEP_MIN);
  else rs = rs.filter((r) => minsLeft(r) <= -KEEP_MIN || (r.hasResult && minsLeft(r) <= 0)).reverse();
  return rs;
}

function leftText(r) {
  if (r.hasResult) return '確定';
  const m = minsLeft(r);
  if (m <= 0) return '締切';
  if (m < 60) return `あと${Math.ceil(m)}分`;
  return `あと${Math.floor(m / 60)}時間${Math.ceil(m % 60)}分`;
}

function exactaResult(r) {
  const ex = r.payouts.exacta || [];
  return ex.length ? ex : null;
}
function topHit(r, pr) {
  const ex = (exactaResult(r) || []).map((x) => x.combo);
  const winner = r.boats.find((b) => b.place === 1);
  const best = pr.probs.win.indexOf(Math.max(...pr.probs.win));
  return { win: winner ? pr.boats[best].waku === winner.waku : null, ex1: ex.length ? ex.includes(pr.exacta[0].combo) : null };
}
function betResult(r, x) {
  if (x.state !== 'buy' || !r.hasResult) return null;
  const ex = exactaResult(r) || [];
  const h = x.picks.map((p) => ex.find((e) => e.combo === p.combo)).find(Boolean);
  return { hit: !!h, amount: h ? h.amount : 0, cost: x.picks.length * 100 };
}

function raceRow(r) {
  const pr = state.preds.get(r.id);
  const open = state.open.has(r.id);
  const left = leftText(r);
  const soon = !r.hasResult && minsLeft(r) > 0 && minsLeft(r) <= 10;
  let strip = '<span class="strip"></span>';
  let picks = '';
  if (pr) {
    const segs = pr.boats.map((b, i) => ({ w: b.waku, p: pr.probs.win[i] }));
    strip = `<span class="strip" aria-label="1着の確率">${segs.map((s) => `<span class="w${s.w}" style="width:${(s.p * 100).toFixed(2)}%">${s.p >= 0.07 ? s.w : ''}</span>`).join('')}</span>`;
    const order = pr.boats.map((b, i) => [b.waku, pr.probs.win[i]]).sort((a, b) => b[1] - a[1]);
    const marks = order.slice(0, 4).map(([w], i) => `${MARKS[i]}${w}`).join(' ');
    const t = pr.exacta[0];
    const xb = exactaPicks(r, pr);
    const br = betResult(r, xb);
    const betTag = xb.state === 'buy' ? `<span class="bet buy">買い ${formation(xb.picks.map((p) => p.combo)).join(' ')} ${buyText(xb.picks.length)}</span>${br ? `<span class="hitmark ${br.hit ? 'yes' : 'no'}">${br.hit ? `的中 +${yen(br.amount * perPoint(xb.picks.length) / 100 - perPoint(xb.picks.length) * xb.picks.length)}` : '外れ'}</span>` : ''}`
      : xb.state === 'pass' ? '<span class="bet pass">見送り</span>' : '';
    const pv = r.preview === 'none' ? '<span class="pv none">展示前</span>' : '';
    picks = `<span class="picks"><span class="marks">${marks}</span><span>本命 <b class="num">${t.combo}</b> ${pct(t.p, 1)}%</span>${betTag}${pv}</span>`;
  } else {
    picks = '<span class="picks">予想できません</span>';
  }
  return `<article class="race" id="r-${r.id}">
    <button type="button" class="race-head" aria-expanded="${open}" data-id="${r.id}">
      <span class="rh-time"><span class="t">${r.closedAt || '--:--'}</span><span class="r">${r.rno}R</span></span>
      <span class="rh-name"><span class="v">${esc(r.venue)}</span><span class="sub">${esc(r.subtitle || '')}</span></span>
      <span class="rh-left${soon ? ' soon' : ''}" data-deadline="${r.id}">${left}</span>
      ${strip}
      ${picks}
    </button>
    ${open ? `<div class="race-body">${raceDetail(r, pr)}</div>` : ''}
  </article>`;
}

function fx(pr, i, name) { const v = pr.X[i][FEATURE_INDEX[name]]; return Number.isFinite(v) ? v : null; }
const fmtST = (v) => (v === null || v === undefined ? '-' : v < 0 ? `F${Math.abs(v).toFixed(2).slice(1)}` : v.toFixed(2).slice(v < 1 ? 1 : 0));

function raceDetail(r, pr) {
  if (!pr) return '';
  const cond = [];
  if (r.weather) cond.push(`天候 <b>${WEATHER[r.weather] || '-'}</b>`);
  if (r.wind !== null && r.wind !== undefined) cond.push(`風 <b>${r.wind}m</b>${r.windDir && r.windDir !== 17 ? ` ${WIND[r.windDir]}` : ''}`);
  if (r.wave !== null && r.wave !== undefined) cond.push(`波 <b>${r.wave}cm</b>`);
  if (r.waterTemp) cond.push(`水温 ${r.waterTemp}℃`);
  cond.push(r.preview === 'none' ? '展示前' : '直前情報 反映済み');
  const boats = pr.boats;
  const rows = r.boats.map((b) => {
    const i = boats.findIndex((x) => x.waku === b.waku);
    if (i < 0) return `<div class="boat absent">${chip(b.waku)}<div><div class="bn">${esc(b.name)}<small>欠場</small></div></div><div></div></div>`;
    const exRank = fx(pr, i, 'exTimeRank');
    const stRank = fx(pr, i, 'exSTRank');
    const course = pr.course[i];
    const pre = [];
    if (r.preview !== 'none') {
      pre.push(`進入<span class="${course !== b.waku ? 'warn' : 'hl'}">${course}</span>`);
      pre.push(`展示ST <span class="${b.exST !== null && b.exST < 0 ? 'warn' : stRank === 1 ? 'hl' : ''}">${fmtST(b.exST)}</span>`);
      pre.push(`展示 <span class="${exRank === 1 ? 'hl' : ''}">${b.exTime !== null ? b.exTime.toFixed(2) : '-'}</span>${exRank ? `(${exRank}位)` : ''}`);
      if (b.tilt !== null && b.tilt !== undefined) pre.push(`チルト${b.tilt > 0 ? '+' : ''}${b.tilt}`);
      if (b.parts && b.parts.length) pre.push(`<span class="warn">部品交換 ${esc(b.parts.join('・'))}</span>`);
      if (b.propNew) pre.push('<span class="warn">新プロペラ</span>');
      if (pr.exhibit) {
        const ex = pr.exhibit[i]; const exMax = Math.max(...pr.exhibit);
        pre.push(`展示評価 <span class="${ex === exMax ? 'hl' : ''}">${pct(ex)}%</span>`);
      }
    }
    const f2 = (v) => (v === null || v === undefined ? '-' : v.toFixed(2));
    const stats = [`勝率 ${f2(b.natWin)}`, `当地 ${f2(b.locWin)}`, `モーター ${b.motor2 !== null ? b.motor2.toFixed(1) : '-'}%`, `平均ST ${fmtST(b.avgST)}`];
    if (b.F) stats.push(`<span class="warn">F${b.F}</span>`);
    const p1 = pr.probs.win[i], p2 = pr.probs.top2[i];
    const bar = (label, p) => `<div class="bar"><span>${label}</span><span class="track"><i style="width:${(p * 100).toFixed(1)}%"></i></span></div>`;
    return `<div class="boat">${chip(b.waku)}
      <div><div class="bn">${esc(b.name)}<small>${RANK_LABEL[b.rank] || ''} ${b.age ?? ''}歳 ${esc(b.branch)}</small></div>
        ${pre.length ? `<div class="bx">${pre.map((x) => `<span class="nb">${x}</span>`).join(' ')}</div>` : ''}
        <div class="bx">${stats.map((x) => `<span class="nb">${x}</span>`).join(' ')}</div></div>
      <div class="bp"><div class="big">${pct(p1)}<small>%</small></div>
        <div class="bars">${bar('1着', p1)}${bar('2連', p2)}</div></div>
    </div>`;
  }).join('');

  // 2連単: 確率の高い順（オッズがあれば期待値）
  const paid = (exactaResult(r) || []).map((x) => x.combo);
  const om = r.odds && r.odds.e ? exactaMap(r.odds.e) : null;
  const bets = pr.exacta.slice(0, 10).map((t) => {
    const hit = paid.includes(t.combo);
    const o = om && om[t.combo];
    const right = o ? `<b>${o.toFixed(1)}</b>倍 期待値<b class="${o * t.p >= 1 ? 'good' : ''}">${(o * t.p).toFixed(2)}</b>` : `目安 <b>${(1 / t.p).toFixed(1)}</b>倍`;
    return `<div class="${hit ? 'hit' : ''}">${comboHtml(t.combo)}</div><div class="p ${hit ? 'hit' : ''}">${pct(t.p, 1)}%</div><div class="o ${hit ? 'hit' : ''}">${right}</div>`;
  }).join('');
  const oddsAt = om ? `オッズ ${r.odds.site ? esc(r.odds.site) : r.odds.at ? hhmm(r.odds.at) : ''}時点${r.odds.final ? '（締切1分前）' : ''}` : '目安＝つり合いの倍率';

  let result = '';
  if (r.hasResult) {
    const order = r.boats.filter((b) => b.place).sort((a, b) => a.place - b.place).map((b) => b.waku);
    const pay = (exactaResult(r) || [])[0];
    result = `<h3>結果</h3><div class="result">${comboHtml(order.slice(0, 2).join('-'))}${pay ? `<span>2連単 <span class="pay">${pay.amount.toLocaleString()}</span>円</span>` : ''}</div>`;
  }

  return `<div class="cond">${cond.map((c) => `<span>${c}</span>`).join('')}</div>
    ${evHtml(r, pr)}
    <h3>2連単の確率<small>${oddsAt}</small></h3>
    <div class="bets">${bets}</div>
    ${rows}
    ${tenkaiHtml(pr)}
    <h3>予想の根拠</h3>
    <ul class="reasons">${reasons(r, pr).map((x) => `<li>${x}</li>`).join('')}</ul>
    ${result}`;
}

// 展開: 誰が・どの決まり手で勝つか、その展開での2着
function tenkaiHtml(pr) {
  if (!pr.scenarios || !pr.scenarios.length) return '';
  const top = pr.scenarios.slice(0, 5);
  const max = top[0].p;
  const rows = top.map((s) => `<div class="sc">
      <div class="sc-h">${chip(s.waku)}<span class="sc-n">${s.course}コース <b>${esc(s.name)}</b></span><span class="sc-p num">${pct(s.p)}%</span></div>
      <div class="sc-bar"><i style="width:${(s.p / max * 100).toFixed(1)}%"></i></div>
      <div class="sc-f">2着 ${s.follow.map((f) => `${chip(f.waku)}<span class="p">${pct(f.p)}%</span>`).join('　')}</div>
    </div>`).join('');
  return `<h3>展開の予想</h3>${rows}`;
}

// 2連単の買い目: 学習で選んだ決め方（model.bet）で、締切前の2連単オッズから選ぶ。確率にオッズは混ぜない
const DEFAULT_BET = { t: 1.2, minp: 0.02, maxo: 60, k: 2, edge: 1, alpha: 1 };
function betRule() {
  const b = state.model && state.model.bet;
  return b ? { rule: { ...b.rule, alpha: 1 }, info: b } : { rule: DEFAULT_BET, info: null };
}
function exactaPicks(r, pr) {
  if (!pr || r.preview === 'none' || !pr.base) return { state: 'none' };
  if (!r.odds || !r.odds.e) return { state: 'wait' };
  const res = evPicks(pr, r, betRule().rule, 'exacta');
  if (!res) return { state: 'wait' };
  return { state: res.picks.length ? 'buy' : 'pass', ...res };
}
function ruleText(rule) {
  return `期待値${rule.t}以上${rule.minp ? `・確率${pct(rule.minp)}%以上` : ''}${rule.edge > 1 ? `・市場の${rule.edge}倍以上` : ''}・${rule.mino ? `${rule.mino}倍超〜` : ''}${rule.maxo}倍${rule.mino ? '' : '以下'}・${rule.k}点`;
}
function evHtml(r, pr) {
  if (r.preview === 'none' || !pr.base) return '';
  const { rule, info } = betRule();
  const past = info ? `<small>検証 回収${pct(info.hold.roi)}%・的中${pct(info.hold.hit, 1)}%</small>` : '';
  const x = exactaPicks(r, pr);
  let body;
  if (x.state === 'buy') {
    const combos = x.picks.map((p) => p.combo);
    const br = betResult(r, x);
    const hit = br ? `<span class="hitmark ${br.hit ? 'yes' : 'no'}">${br.hit ? `的中 払戻${yen(br.amount * perPoint(combos.length) / 100)}` : '外れ'}</span>` : '';
    body = `<div class="ev-sum"><b class="num big-bet">${formation(combos).join('　')}</b><b class="num big-bet">${buyText(combos.length)}</b>${x.picks.map((p) => `<span>${pct(p.p, 1)}%・${p.odds.toFixed(1)}倍・期待値 <b class="good">${p.ev.toFixed(2)}</b></span>`).join('')} ${hit}</div>`;
  } else if (x.state === 'pass') {
    body = '<div class="ev-sum"><b>見送り</b></div>';
  } else {
    const list = pr.exacta.filter((p) => p.p >= (rule.minp || 0)).slice(0, 3);
    body = `<div class="ev-sum">オッズ待ち　${list.map((p) => `${comboHtml(p.combo)} <b class="num">${Math.max(rule.t / p.p, (rule.mino || 0) + 0.1, 1).toFixed(1)}</b>倍以上で買い`).join('　')}</div>`;
  }
  return `<div class="rec main-bet"><div class="rec-h"><b>2連単の買い目</b>${past}</div>${body}<div class="rule">${esc(ruleText(rule))}</div></div>`;
}

// 根拠: 特徴量の値から、目立つ点を文章にする
function reasons(r, pr) {
  const out = [];
  const B = pr.boats;
  const idx = B.map((_, i) => i).sort((a, b) => pr.probs.win[b] - pr.probs.win[a]);
  const name = (i) => `${chip(B[i].waku)} ${esc((B[i].name || '').split(/\s/)[0])}`;
  const c1 = pr.course.indexOf(1);
  if (r.preview !== 'none') {
    const moved = B.map((b, i) => [b, pr.course[i]]).filter(([b, c]) => c !== b.waku);
    if (moved.length) out.push(`進入が枠と違う: ${moved.map(([b, c]) => `${chip(b.waku)}→${c}コース`).join('　')}`);
    const exBest = B.findIndex((_, i) => fx(pr, i, 'exTimeRank') === 1);
    if (exBest >= 0) out.push(`展示タイム1位 ${name(exBest)}（${B[exBest].exTime?.toFixed(2)}）`);
    const stBest = B.findIndex((_, i) => fx(pr, i, 'exSTRank') === 1);
    if (stBest >= 0) out.push(`スタート展示で一番早い ${name(stBest)}（${fmtST(B[stBest].exST)}）`);
    const flying = B.filter((b) => b.exST !== null && b.exST < 0);
    if (flying.length) out.push(`スタート展示でフライング: ${flying.map((b) => chip(b.waku)).join(' ')}`);
  }
  if (c1 >= 0) {
    const vc = fx(pr, c1, 'vcWin'); const own = fx(pr, c1, 'pCourseWin');
    if (vc !== null) out.push(`${esc(r.venue)}の1コース1着率 ${pct(vc)}%${own !== null ? `／${name(c1)}の1コース1着率 ${pct(own)}%` : ''}`);
  }
  const top = idx[0];
  const nw = fx(pr, top, 'natWinRank'); const mr = fx(pr, top, 'motor2Rank'); const sa = fx(pr, top, 'pSetsuAvg');
  const good = [];
  if (nw === 1) good.push('勝率がレース内1位');
  if (mr === 1) good.push('モーター2連率がレース内1位');
  if (sa !== null && fx(pr, top, 'pSetsuN') >= 2 && sa <= 2.5) good.push(`今節の平均着順 ${sa.toFixed(1)}`);
  if (good.length) out.push(`本命 ${name(top)}: ${good.join('、')}`);
  for (const i of idx.slice(1, 3)) {
    const sv = fx(pr, i, 'exSTvsInner');
    if (sv !== null && sv <= -0.05 && pr.course[i] >= 2) out.push(`${name(i)} は内側の艇より展示STが ${Math.abs(sv).toFixed(2)} 早い`);
  }
  const fs = B.filter((b) => b.F > 0);
  if (fs.length) out.push(`F持ち: ${fs.map((b) => chip(b.waku)).join(' ')}`);
  return out;
}

function renderScore(rs) {
  const done = rs.filter((r) => r.hasResult && state.preds.get(r.id) && exactaResult(r));
  if (!done.length) return '';
  let win = 0, ex1 = 0, bets = 0, ret = 0, hits = 0;
  for (const r of done) {
    const pr = state.preds.get(r.id);
    const h = topHit(r, pr);
    win += h.win ? 1 : 0; ex1 += h.ex1 ? 1 : 0;
    const br = betResult(r, exactaPicks(r, pr));
    if (br) { bets += br.cost / 100; ret += br.amount; hits += br.hit ? 1 : 0; }
  }
  return `<section class="score"><h2>答え合わせ（${done.length}レース）</h2><div class="grid">
    <div><div class="k">1着の的中</div><div class="v">${pct(win / done.length)}<small>%</small></div></div>
    <div><div class="k">2連単 本命の的中</div><div class="v">${pct(ex1 / done.length)}<small>%</small></div></div>
    <div><div class="k">買い目 ${bets}点 的中${hits}（1点${yen(stake())}）</div><div class="v">${bets ? pct(ret / (bets * 100)) : '-'}<small>%回収</small></div><div class="k">${bets ? `${ret / 100 * stake() - bets * stake() >= 0 ? '+' : ''}${yen(ret / 100 * stake() - bets * stake())}` : ''}</div></div>
  </div></section>`;
}

// 毎日の記録（その日の朝の時点の近況とモデルで予想し直した成績）
const REC_KEY = '2連単 期待値の買い目（メイン）';
function renderRecord() {
  const days = ((state.report && state.report.days) || []).filter((d) => d.rec && d.rec[REC_KEY]);
  if (!days.length) return '';
  const recent = days.slice(-14).reverse();
  const tot = days.slice(-30).reduce((a, d) => { const x = d.rec[REC_KEY]; a.b += x.bets; a.h += x.hits; a.ret += x.ret; return a; }, { b: 0, h: 0, ret: 0 });
  const rows = recent.map((d) => { const x = d.rec[REC_KEY]; return `<tr><td>${+d.date.slice(5, 7)}/${+d.date.slice(8)}</td><td>${x.bets}</td><td>${x.hits}</td><td>${pct(x.ret / Math.max(x.bets * 100, 1))}%</td><td>${pct((d.ex1 || 0) / d.races)}%</td></tr>`; }).join('');
  return `<section class="score"><h2>毎日の記録（直近${Math.min(days.length, 30)}日 ${tot.b}点 回収${pct(tot.ret / Math.max(tot.b * 100, 1))}%）</h2>
    <table class="rec"><thead><tr><th>日付</th><th>買い目</th><th>的中</th><th>回収</th><th>本命的中</th></tr></thead><tbody>${rows}</tbody></table></section>`;
}

// 仮想の損益: 入金額から始めて、採用した買い方を機械的に買い続けた場合の残高（1点＝残高の1/200、100円単位、上限3,000円）
function simulate(start) {
  const pts = []; let bank = start; let peak = start; let dd = 0; let bets = 0; let cost = 0; let ret = 0; let last = '';
  const push = (d, seg) => { if (d !== last) { pts.push({ d, bank, seg }); last = d; } else pts[pts.length - 1].bank = bank; };
  pts.push({ d: '', bank, seg: 1 });
  const step = (n, payPer100, d, seg) => {
    const st = Math.min(MAX_STAKE, Math.floor(bank / 200 / 100) * 100);
    if (st < 100) return;
    for (let i = 0; i < n; i++) { bank -= st; cost += st; bets++; }
    bank += st * payPer100; ret += st * payPer100;
    peak = Math.max(peak, bank); dd = Math.max(dd, peak - bank);
    push(d, seg);
  };
  // 選ぶのに使った期間（開発）は成績が良く出るので含めない。確かめた期間（本番テスト）から始める
  for (const [d, seg, , pay] of (state.sim && state.sim.bets) || []) if (seg === 1) step(1, pay, d, seg);
  const days = ((state.report && state.report.days) || []).filter((x) => x.rec && x.rec[REC_KEY]);
  const lastSim = pts.length > 1 ? pts[pts.length - 1].d : '';
  for (const x of days) {
    const d = x.date.replace(/-/g, '');
    if (d <= lastSim) continue;
    const r = x.rec[REC_KEY];
    if (!r.bets) continue;
    // 1日分をまとめて: 点数ぶん買って、払戻の合計（100円あたり）
    const st = Math.min(MAX_STAKE, Math.floor(bank / 200 / 100) * 100);
    if (st < 100) break;
    bank += st * (r.ret / 100 - r.bets); cost += st * r.bets; ret += st * r.ret / 100; bets += r.bets;
    peak = Math.max(peak, bank); dd = Math.max(dd, peak - bank);
    push(d, 2);
  }
  return { pts, bank, dd, bets, roi: cost ? ret / cost : 0 };
}
function renderSim() {
  if (!state.sim) return '';
  const start = state.bank;
  const S = simulate(start);
  if (S.pts.length < 3) return '';
  const W = 340, H = 170, L = 4, T = 8, B = 18;
  const vals = S.pts.map((p) => p.bank);
  const lo = Math.min(start, ...vals), hi = Math.max(start, ...vals);
  const x = (i) => L + (W - L * 2) * i / (S.pts.length - 1);
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / Math.max(hi - lo, 1));
  const segs = [[], [], []];
  for (let i = 1; i < S.pts.length; i++) { const k = S.pts[i].seg; if (!segs[k].length) segs[k].push(i - 1); segs[k].push(i); }
  segs.forEach((a, k) => { segs[k] = a.map((i) => `${x(i).toFixed(1)},${y(S.pts[i].bank).toFixed(1)}`); });
  const cls = ['sim-dev', 'sim-hold', 'sim-live'];
  const lines = segs.map((pp, k) => (pp.length > 1 ? `<polyline class="${cls[k]}" points="${pp.join(' ')}"/>` : '')).join('');
  const lab = (i) => { const d = S.pts[i].d; return d ? `${+d.slice(4, 6)}/${+d.slice(6, 8)}` : ''; };
  const xt = [1, Math.floor(S.pts.length / 2), S.pts.length - 1].map((i) => `<text x="${x(i).toFixed(1)}" y="${H - 4}" text-anchor="${i === 1 ? 'start' : i === S.pts.length - 1 ? 'end' : 'middle'}">${lab(i)}</text>`).join('');
  const gain = S.bank - start;
  return `<section class="score"><h2>仮想の損益（入金${yen(start)}から機械的に買い続けた場合）</h2>
    <div class="grid">
      <div><div class="k">残高</div><div class="v">${Math.round(S.bank / 1000).toLocaleString()}<small>千円</small></div><div class="k">${gain >= 0 ? '+' : ''}${yen(gain)}</div></div>
      <div><div class="k">回収率（${S.bets.toLocaleString()}点）</div><div class="v">${pct(S.roi)}<small>%</small></div></div>
      <div><div class="k">最大の落ち込み</div><div class="v">${Math.round(S.dd / 1000).toLocaleString()}<small>千円</small></div></div>
    </div>
    <svg class="sim" viewBox="0 0 ${W} ${H}" role="img" aria-label="残高の推移"><line class="base" x1="${L}" x2="${W - L}" y1="${y(start).toFixed(1)}" y2="${y(start).toFixed(1)}"/>${lines}${xt}</svg>
    <div class="legend"><span class="sim-hold">検証 ${+(state.sim.hold || ['0-1'])[0].slice(5)}〜${+(state.sim.hold || ['0-1']).slice(-1)[0].slice(5)}月（確定オッズ）</span><span class="sim-live">実際の記録（締切前オッズ）</span></div>
  </section>`;
}

// 2連単の買い方の比較（research_exacta.json）
function renderResearch() {
  const E = state.exa;
  if (!E) return '<div class="note">比較の結果はまだありません。</div>';
  const pt = (E.prob_table || []).filter((x) => x.gamma === 1 || x.name.startsWith('市場'));
  const nameOf = (n) => ({ tenkai: '展開モデル（標準）', tenkai_big: '展開モデル（大きめ）', gbm: '機械学習', clogit: '条件付きロジット', ce: '枠＋展示航走だけ' }[n] || n);
  const probRows = pt.map((x) => `<tr class="${x.name === E.best.prob ? 'chosen' : ''}"><td class="l">${esc(nameOf(x.name))}${x.name === E.best.prob ? '<br><b>← 採用</b>' : ''}</td><td>${x.dev_ll.toFixed(3)}</td><td>${x.hold_ll.toFixed(3)}</td><td>${pct(x.hold_top1, 1)}</td></tr>`).join('');
  const seen = new Set();
  const top = (E.top || []).filter((c) => { const k = `${c.prob}|${c.gamma}|${c.hold.bets}|${c.hold.roi}`; if (seen.has(k)) return false; seen.add(k); return true; }).slice(0, 8);
  const rows = top.map((c) => `<tr class="${c === E.top[0] ? 'chosen' : ''}"><td class="l">${esc(c.label.replace('2連単 ', ''))}<br><small>${esc(nameOf(c.prob))}${c.gamma !== 1 ? ` γ${c.gamma}` : ''}</small></td><td>${pct(c.dev.roi)}%</td><td>${pct(c.hold.roi)}%<br><small>±${pct(c.hold.se)}</small></td><td>${pct(c.hold.hit, 1)}%</td><td>${(c.hold.bets / Math.max(1, E.hold_days || 120)).toFixed(0)}</td></tr>`).join('');
  const months = Object.entries(E.best.hold_months || {}).map(([m, v]) => `${+m.slice(5)}月 ${pct(v)}%`).join('・');
  return `<section class="score research"><h2>2連単の買い方の比較（${esc(E.when || '')}）</h2>
    <p class="explain">開発 ${esc((E.dev || []).join('・'))} で選び、本番 ${esc((E.hold || []).join('・'))} で確認（確定オッズ）。</p>
    <h3>確率の当たり具合</h3>
    <table class="rec-t"><thead><tr><th class="l">予想</th><th>開発<br>logloss</th><th>本番<br>logloss</th><th>本番<br>本命的中</th></tr></thead><tbody>${probRows}</tbody></table>
    <h3>買い方</h3>
    <table class="rec-t"><thead><tr><th class="l">買い方</th><th>開発<br>回収</th><th>本番<br>回収</th><th>的中</th><th>1日の<br>点数</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="explain">採用した買い方の月別: ${months}</p></section>`;
}

function render() {
  renderStatus();
  const venues = [...new Map(state.races.map((r) => [r.jcd, r.venue])).entries()].sort((a, b) => a[0] - b[0]);
  $('#venues').innerHTML = state.view === 'research' ? '' : [[0, 'すべて'], ...venues].map(([j, v]) => `<button type="button" data-venue="${j}" aria-pressed="${state.venue === j}">${esc(v)}</button>`).join('');
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
  document.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
  $('#modeRow').hidden = state.view === 'research' || !(state.model && state.model.ce);
  $('#bankRow').hidden = state.view === 'research';
  const list = $('#list');
  if (state.view === 'research') { list.innerHTML = renderResearch(); renderFooter(); return; }
  if (state.error && !state.races.length) { list.innerHTML = `<div class="note bad">${esc(state.error)}</div>`; return; }
  if (!state.races.length) { list.innerHTML = state.loading ? '' : '<div class="note">この日のレースはありません。</div>'; return; }
  const rs = filtered();
  let html = state.error ? `<div class="note bad">${esc(state.error)}</div>` : '';
  if (state.view === 'done') html += renderScore(rs) + renderSim() + renderRecord();
  if (!rs.length) html += `<div class="note">${state.view === 'upcoming' ? 'これから締切のレースはありません。' : 'まだ終わったレースはありません。'}</div>`;
  html += rs.map(raceRow).join('');
  list.innerHTML = html;
  renderFooter();
}

function renderFooter() {
  const m = state.model;
  if (!m) return;
  const lines = [];
  lines.push(`予想モデル: ${m.trained_until || '-'} まで学習。${m.bet ? `2連単の買い方: ${esc(ruleText(m.bet.rule))}（検証 回収${pct(m.bet.hold.roi)}%）` : ''}`);
  lines.push('予想は参考情報です。舟券の購入は20歳以上、余裕のある範囲で。');
  $('#foot').innerHTML = lines.map((x) => `<p>${x}</p>`).join('');
}

// ------------------------------------------------------------------ 操作
document.addEventListener('click', (e) => {
  const head = e.target.closest('.race-head');
  if (head) {
    const id = head.dataset.id;
    if (state.open.has(id)) state.open.delete(id); else state.open.add(id);
    const r = state.races.find((x) => x.id === id);
    const art = document.getElementById(`r-${id}`);
    if (r && art) art.outerHTML = raceRow(r);
    return;
  }
  const body = e.target.closest('.race-body');
  if (body && !e.target.closest('a, button, input')) {
    const art = body.closest('.race');
    const id = art.id.slice(2);
    state.open.delete(id);
    const r = state.races.find((x) => x.id === id);
    if (r) art.outerHTML = raceRow(r);
    const el = document.getElementById(`r-${id}`);
    if (el && el.getBoundingClientRect().top < 0) el.scrollIntoView({ block: 'start' });
    return;
  }
  const vb = e.target.closest('[data-venue]');
  if (vb) { state.venue = +vb.dataset.venue; render(); return; }
  const mb = e.target.closest('[data-mode]');
  if (mb) {
    state.mode = mb.dataset.mode; savePref('mode2', state.mode);
    document.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
    predictAll().then((ok) => { if (ok) render(); });
    return;
  }
  const tb = e.target.closest('[data-view]');
  if (tb) { state.view = tb.dataset.view; render(); window.scrollTo({ top: 0 }); return; }
});
$('#prev').addEventListener('click', () => { state.date = addDays(state.date, -1); state.races = []; state.open.clear(); state.view = 'done'; refresh(true); });
$('#next').addEventListener('click', () => {
  const n = addDays(state.date, 1);
  if (n > addDays(jstToday(), 1)) return;
  state.date = n; state.races = []; state.open.clear(); state.view = n >= jstToday() ? 'upcoming' : 'done'; refresh(true);
});
$('#reload').addEventListener('click', () => refresh(false));
function renderBank() { $('#bank').value = state.bank; $('#stake').textContent = yen(stake()); }
$('#bank').addEventListener('change', (e) => {
  const v = Math.max(0, Math.round(+e.target.value || 0));
  state.bank = v || 20000; savePref('bank', String(state.bank)); renderBank(); render();
});
renderBank();

// 締切直前のオッズ: 開いているレースが締切3分前〜締切のあいだは、15秒ごとに GitHub から直接読む（公開の反映を待たない）
const LIVE_API = 'https://api.github.com/repos/koukikoukisanosano-ai/boatrace-yosou/contents/docs/live';
let fastBusy = false;
const fastCache = new Map();
async function fastOdds() {
  if (fastBusy || document.visibilityState !== 'visible' || state.date !== jstToday()) return;
  const near = state.races.filter((r) => state.open.has(r.id) && minsLeft(r) <= 3 && minsLeft(r) > -0.5 && !(r.odds && r.odds.final));
  if (!near.length) return;
  fastBusy = true;
  try {
    const res = await fetch(`${LIVE_API}/${compact(state.date)}.json?ref=main&t=${Date.now()}`, { headers: { Accept: 'application/vnd.github.raw' }, cache: 'no-store' });
    if (!res.ok) return;
    const live = await res.json();
    let changed = false;
    for (const r of near) {
      const o = live.odds && live.odds[`${r.jcd}-${r.rno}`];
      if (!o || !(o.t || o.e)) continue;
      const at = o.at ? new Date(o.at) : null;
      if (r.odds && r.odds.at && at && +r.odds.at === +at) continue;
      r.odds = { t: o.t || null, e: o.e || null, at, site: o.site || '', final: !!o.final };
      fastCache.set(`${r.jcd}-${r.rno}`, r.odds);
      changed = true;
    }
    if (changed) {
      await predictAll();
      for (const r of near) { const art = document.getElementById(`r-${r.id}`); if (art) art.outerHTML = raceRow(r); }
    }
  } catch (e) { console.error(e); } finally { fastBusy = false; }
}
setInterval(fastOdds, 15000);

setInterval(() => {
  if (document.visibilityState !== 'visible') return;
  if (state.date === jstToday() || state.date === addDays(jstToday(), 1)) refresh(false);
}, REFRESH_MS);
setInterval(() => {
  document.querySelectorAll('[data-deadline]').forEach((el) => {
    const r = state.races.find((x) => x.id === el.dataset.deadline);
    if (r) { el.textContent = leftText(r); el.classList.toggle('soon', !r.hasResult && minsLeft(r) > 0 && minsLeft(r) <= 10); }
  });
}, 15000);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && state.loadedAt && Date.now() - state.loadedAt > 30000) refresh(false); });

refresh(true);
