// 予想ページ: データを読み込んで、スマホの中で予想を計算して表示する
import { normalizeDay, prepareModel, predictRace, Snapshot, previewState, oddsMap, exactaMap, evPicks, formation, FEATURE_INDEX, RANK_LABEL } from './core.js';

const OPENAPI = 'https://boatraceopenapi.github.io/api/v1';
const REFRESH_MS = 60 * 1000;
const WIND = ['', '北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東', '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西', '無風'];
const WEATHER = { 1: '晴', 2: '曇り', 3: '雨', 4: '雪', 5: '台風', 6: '霧' };
const MARKS = ['◎', '○', '▲', '△'];

const state = {
  date: jstToday(), model: null, snap: null, snapNote: '', races: [], preds: new Map(),
  openApiAt: null, liveAt: null, liveCount: 0, view: 'upcoming', venue: 0, open: new Set(),
  error: '', liveError: false, loading: false, loadedAt: null, report: null, snapExact: true, research: null, ev: null,
  mode: loadPref('mode', 'exhibit'),
};
function loadPref(k, d) { try { return localStorage.getItem(k) || d; } catch (e) { return d; } }
function savePref(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 保存できなくても動く */ } }
// 予想の重み: 展示最重視（枠＋展示航走のモデルを強く混ぜる）/ 総合（的中率がいちばん高い混ぜ方）
const exhibitW = () => (state.mode === 'total' ? 0 : undefined);

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
    if (o && o.t) r.odds = { t: o.t, at: o.at ? new Date(o.at) : null, site: o.site || '' };
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
    if (!state.research) { try { state.research = (await getJSON('data/research.json', 'no-cache')).json; } catch (e) { state.research = null; } }
    if (!state.ev) { try { state.ev = (await getJSON('data/research_ev.json', 'no-cache')).json; } catch (e) { state.ev = null; } }
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
  let t = performance.now();
  for (const r of order) {
    const key = JSON.stringify([state.mode, r.preview, r.odds ? r.odds.site || String(r.odds.at) : '', r.boats.map((b) => [b.exCourse, b.exST, b.exTime, b.tilt, b.weightAdj, b.propNew, (b.parts || []).length, b.place ?? null]), r.wind, r.wave, r.windDir]);
    if (predKey.get(r.id) === key && state.preds.has(r.id)) { keep.set(r.id, state.preds.get(r.id)); continue; }
    try { const p = predictRace(r, state.model, state.snap, { exhibitW: exhibitW() }); if (p) { keep.set(r.id, p); predKey.set(r.id, key); } } catch (e) { console.error(r.id, e); }
    if (performance.now() - t > 40) {
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

function filtered() {
  let rs = state.races.filter((r) => state.venue === 0 || r.jcd === state.venue);
  if (state.view === 'upcoming') rs = rs.filter((r) => !r.hasResult && minsLeft(r) > -3);
  else rs = rs.filter((r) => r.hasResult || minsLeft(r) <= -3).reverse();
  return rs;
}

function leftText(r) {
  if (r.hasResult) return '確定';
  const m = minsLeft(r);
  if (m <= 0) return '締切';
  if (m < 60) return `あと${Math.ceil(m)}分`;
  return `あと${Math.floor(m / 60)}時間${Math.ceil(m % 60)}分`;
}

function topHit(r, pr) {
  const tp = (r.payouts.trifecta || []).map((x) => x.combo);
  const winner = r.boats.find((b) => b.place === 1);
  const best = pr.probs.win.indexOf(Math.max(...pr.probs.win));
  return {
    win: winner ? pr.boats[best].waku === winner.waku : null,
    tri5: tp.length ? pr.trifecta.slice(0, 5).some((t) => tp.includes(t.combo)) : null,
    tri1: tp.length ? tp.includes(pr.trifecta[0].combo) : null,
  };
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
    const t = pr.trifecta[0];
    let hit = '';
    if (r.hasResult) {
      const h = topHit(r, pr);
      hit = `<span class="hitmark ${h.win ? 'yes' : 'no'}">1着${h.win ? '的中' : '外れ'}</span><span class="hitmark ${h.tri5 ? 'yes' : 'no'}">3連単5点${h.tri5 ? '的中' : '外れ'}</span>`;
    }
    const pv = r.preview === 'none' ? '<span class="pv none">展示前</span>' : '<span class="pv full">直前反映</span>';
    picks = `<span class="picks"><span class="marks">${marks}</span><span>本命 <b class="num">${t.combo}</b> ${pct(t.p, 1)}%</span>${r.hasResult ? hit : pv}</span>`;
  } else {
    picks = '<span class="picks">予想できません（出走表が不足）</span>';
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
  if (!pr) return '<p class="explain">出走表が足りないため予想できません。</p>';
  const cond = [];
  if (r.weather) cond.push(`天候 <b>${WEATHER[r.weather] || '-'}</b>`);
  if (r.wind !== null && r.wind !== undefined) cond.push(`風 <b>${r.wind}m</b>${r.windDir && r.windDir !== 17 ? ` ${WIND[r.windDir]}` : ''}`);
  if (r.wave !== null && r.wave !== undefined) cond.push(`波 <b>${r.wave}cm</b>`);
  if (r.waterTemp) cond.push(`水温 ${r.waterTemp}℃`);
  let src = '展示前（出走表だけの予想）';
  if (r.preview !== 'none') src = r.previewFrom === 'live' ? `直前情報 反映済み（公式サイトから${r.previewAt ? ` ${hhmm(r.previewAt)}` : ''}取得）` : '直前情報 反映済み';
  const boats = pr.boats;
  const all = r.boats;
  const rows = all.map((b) => {
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
    const p1 = pr.probs.win[i], p2 = pr.probs.top2[i], p3 = pr.probs.top3[i];
    const bar = (label, p) => `<div class="bar"><span>${label}</span><span class="track"><i style="width:${(p * 100).toFixed(1)}%"></i></span></div>`;
    return `<div class="boat">${chip(b.waku)}
      <div><div class="bn">${esc(b.name)}<small>${RANK_LABEL[b.rank] || ''} ${b.age ?? ''}歳 ${esc(b.branch)}</small></div>
        ${pre.length ? `<div class="bx">${pre.map((x) => `<span class="nb">${x}</span>`).join(' ')}</div>` : ''}
        <div class="bx">${stats.map((x) => `<span class="nb">${x}</span>`).join(' ')}</div></div>
      <div class="bp"><div class="big">${pct(p1)}<small>%</small></div>
        <div class="bars">${bar('1着', p1)}${bar('2連', p2)}${bar('3連', p3)}</div></div>
    </div>`;
  }).join('');

  const tp = (r.payouts.trifecta || []).map((x) => x.combo);
  const top = pr.trifecta.slice(0, 8);
  const om = r.odds ? oddsMap(r.odds.t) : null;
  const bets = top.map((t) => {
    const hit = tp.includes(t.combo);
    const o = om && om[t.combo];
    const right = o ? `<b>${o.toFixed(1)}</b>倍 期待値<b class="${o * t.p >= 1 ? 'good' : ''}">${(o * t.p).toFixed(2)}</b>` : `目安 <b>${(1 / t.p).toFixed(1)}</b>倍`;
    return `<div class="${hit ? 'hit' : ''}">${comboHtml(t.combo)}</div><div class="p ${hit ? 'hit' : ''}">${pct(t.p, 1)}%</div><div class="o ${hit ? 'hit' : ''}">${right}</div>`;
  }).join('');
  const cum5 = pr.trifecta.slice(0, 5).reduce((a, t) => a + t.p, 0);
  const minor = (list, n, label) => `<div><b>${label}</b> ${list.slice(0, n).map((x) => `${comboHtml(x.combo)}<span class="p">${pct(x.p)}%</span>`).join('　')}</div>`;
  const oddsNote = r.odds
    ? `オッズは締切前の${r.odds.site ? ` ${esc(r.odds.site)}` : r.odds.at ? ` ${hhmm(r.odds.at)}` : ''}時点（公式サイトから自動取得）。予想にもこのオッズを混ぜています。期待値＝確率×オッズ（1より大きいと、予想どおりなら得）。締切直前にオッズは動くので、買う前にアプリで最新のオッズを確認してください。`
    : '目安＝確率から計算したつり合いの倍率。実際のオッズがこれより高ければ、予想どおりなら得な買い目です（締切7〜2分前にオッズを自動取得すると、オッズと期待値を表示します）。';

  let result = '';
  if (r.hasResult) {
    const order = r.boats.filter((b) => b.place).sort((a, b) => a.place - b.place).map((b) => b.waku);
    const pay = (r.payouts.trifecta || [])[0];
    const h = topHit(r, pr);
    result = `<h3>結果</h3><div class="result">${comboHtml(order.slice(0, 3).join('-'))}${pay ? `<span>3連単 <span class="pay">${pay.amount.toLocaleString()}</span>円</span>` : ''}
      <span class="hitmark ${h.win ? 'yes' : 'no'}">1着予想 ${h.win ? '的中' : '外れ'}</span><span class="hitmark ${h.tri5 ? 'yes' : 'no'}">3連単上位5点 ${h.tri5 ? '的中' : '外れ'}</span></div>`;
  }

  return `<div class="cond">${cond.map((c) => `<span>${c}</span>`).join('')}<span>${src}</span></div>
    ${rows}
    ${pr.exhibit ? `<p class="explain">展示評価＝枠（コース）と展示航走（展示タイム・スタート展示・チルト・部品交換など）だけで見た1着の確率。${state.mode === 'total' ? '今は「総合」なので、この評価は予想に少しだけ混ぜています。' : '「展示最重視」では、この評価を予想に強く混ぜています（枠以外では展示航走がいちばん効く重さ）。'}</p>` : ''}
    ${tenkaiHtml(pr)}
    ${evHtml(r, pr)}
    ${recommendHtml(r, pr)}
    <h3>3連単の確率<small>上位5点で ${pct(cum5)}%${pr.withOdds ? '・オッズ反映' : ''}</small></h3>
    <div class="bets">${bets}</div>
    <p class="explain">${oddsNote}</p>
    <h3>ほかの券種</h3>
    <div class="minor">${minor(pr.exacta, 3, '2連単')}${minor(pr.trio, 3, '3連複')}${minor(pr.quinella, 2, '2連複')}${minor(pr.wide, 3, '拡連複')}${minor(pr.place, 2, '複勝')}</div>
    <h3>予想の根拠</h3>
    <ul class="reasons">${reasons(r, pr).map((x) => `<li>${x}</li>`).join('')}</ul>
    ${result}`;
}

// 展開予想: 誰が・どの決まり手で勝つか、その展開での2・3着
function tenkaiHtml(pr) {
  if (!pr.scenarios || !pr.scenarios.length) return '';
  const top = pr.scenarios.slice(0, 5);
  const max = top[0].p;
  const rows = top.map((s) => `<div class="sc">
      <div class="sc-h">${chip(s.waku)}<span class="sc-n">${s.course}コース <b>${esc(s.name)}</b></span><span class="sc-p num">${pct(s.p)}%</span></div>
      <div class="sc-bar"><i style="width:${(s.p / max * 100).toFixed(1)}%"></i></div>
      <div class="sc-f">続く並び ${s.follow.map((f) => `${comboHtml(f.combo)}<span class="p">${pct(f.p)}%</span>`).join('　')}</div>
    </div>`).join('');
  return `<h3>展開の予想<small>選手の組み合わせから、決まり手ごとに着順を予想</small></h3>${rows}
    <p class="explain">展開の確率は展開モデルだけの値（展示評価やオッズを混ぜる前）。「続く並び」はその展開になったときの2・3着の確率です（展開ごとに、1着艇との並び・コースの差・選手の力と展示の差から計算）。</p>`;
}

// 期待値のある買い目（3連単・2連単）: 予想方法の比較（research_ev.json）で選んだ決め方で、締切前オッズから選ぶ
const DEFAULT_EV = { t: 1.2, minp: 0.01, maxo: 60, k: 3, alpha: 0.5 };
function evRule(kind) {
  const E = state.ev;
  const w = state.mode === 'total' ? '0' : String((state.model && state.model.exhibit_w) ?? '');
  const byw = E && E.by_w && (E.by_w[w] || E.by_w[Object.keys(E.by_w).pop()]);
  const best = byw && byw[kind];
  if (!best) return { rule: DEFAULT_EV, info: null };
  return { rule: { ...best.rule, alpha: best.alpha }, info: best };
}
function evHtml(r, pr) {
  if (r.preview === 'none' || !pr.base) return '';
  const blocks = [];
  for (const [kind, name] of [['trifecta', '3連単'], ['exacta', '2連単']]) {
    const { rule, info } = evRule(kind);
    const past = info ? `<small>過去の検証（本番テスト期間）: 回収${pct(info.hold.roi)}%・的中${pct(info.hold.hit, 1)}%・${info.hold.bets.toLocaleString()}点</small>` : '';
    const cond = `期待値${rule.t}以上${rule.minp ? `・確率${pct(rule.minp)}%以上` : ''}・${rule.maxo}倍以下・1レース${rule.k}点まで`;
    const res = evPicks(pr, r, rule, kind);
    let body;
    if (res && res.picks.length) {
      const combos = res.picks.map((x) => x.combo);
      const ptot = res.picks.reduce((a, x) => a + x.p, 0);
      const ret = res.picks.reduce((a, x) => a + x.p * x.odds, 0) / res.picks.length;
      const paid = (r.payouts[kind] || []).map((x) => x.combo);
      const hit = r.hasResult ? `<span class="hitmark ${combos.some((c) => paid.includes(c)) ? 'yes' : 'no'}">${combos.some((c) => paid.includes(c)) ? '的中' : '外れ'}</span>` : '';
      body = `<table class="ev"><tbody>${res.picks.map((x) => `<tr><td>${comboHtml(x.combo)}</td><td>${pct(x.p, 1)}%</td><td>${x.odds.toFixed(1)}倍</td><td class="good">${x.ev.toFixed(2)}</td></tr>`).join('')}</tbody></table>
        <div class="ev-sum">買い方 <b class="num">${formation(combos).join('　')}</b>（${combos.length}点・各100円で${combos.length * 100}円）　当たる確率 計${pct(ptot)}%・期待値 平均${ret.toFixed(2)} ${hit}</div>`;
    } else if (res) {
      const near = res.top.slice(0, 3).map((x) => `${comboHtml(x.combo)} ${x.odds.toFixed(1)}倍 期待値${x.ev.toFixed(2)}`).join('　');
      body = `<div class="ev-sum">条件に合う買い目なし（見送り）。期待値の高い順: ${near}</div>`;
    } else if (kind === 'exacta' && r.odds && r.odds.t && !r.odds.e) {
      body = '<div class="ev-sum">2連単のオッズをまだ取れていません。</div>';
    } else {
      // オッズがまだ無いとき: この倍率以上なら期待値が条件を超える、という目安
      const list = (kind === 'exacta' ? pr.exacta : pr.trifecta).filter((x) => x.p >= (rule.minp || 0)).slice(0, kind === 'exacta' ? 3 : 5);
      body = `<div class="ev-sum">締切7〜2分前にオッズを取ると選びます。目安: ${list.map((x) => `${comboHtml(x.combo)} <b class="num">${(rule.t / x.p).toFixed(1)}</b>倍以上なら買い`).join('　')}</div>`;
    }
    blocks.push(`<div class="rec"><div class="rec-h"><b>${name}</b>${past}</div><div class="explain">${cond}${rule.alpha < 1 ? `（予想${pct(rule.alpha)}%＋オッズ${pct(1 - rule.alpha)}%の確率で計算）` : ''}</div>${body}</div>`);
  }
  return `<h3>期待値のある買い目<small>期待値＝確率×オッズ</small></h3>${blocks.join('')}
    <p class="explain">期待値が1を超える買い目は「予想が正しければ得」な買い目です。ただし過去の検証でも回収率が安定して100%を超えるわけではありません（下の「方法の比較」参照）。締切直前にオッズが動くので、買う前に最新のオッズで期待値を確かめてください。</p>`;
}

// おすすめの買い方: 予想方法の比較（research.json）で、開発期間で選び本番テスト期間で確かめた買い方
const BET_LABEL = { trifecta: '3連単', trio: '3連複', exacta: '2連単', quinella: '2連複', quinella_place: '拡連複', win: '単勝', place: '複勝' };
const BET_LIST = { trifecta: 'trifecta', trio: 'trio', exacta: 'exacta', quinella: 'quinella', quinella_place: 'wide', win: 'win', place: 'place' };
function recommendHtml(r, pr) {
  const recs = (state.research && state.research.recommend) || [];
  // 展開モデルでは確率の出方が違うので、前の予想方法で選んだ買い方は出さない（期待値の買い目を使う）
  if (!recs.length || r.preview === 'none' || (state.model && state.model.tenkai)) return '';
  const rows = [];
  for (const rc of recs) {
    const list = pr[BET_LIST[rc.bet]] || [];
    if (!list.length) continue;
    const conf = rc.rule.conf || 0;
    const ok = list[0].p >= conf;
    const picks = list.slice(0, rc.rule.k);
    const paid = (r.payouts[rc.bet] || []).map((x) => x.combo);
    const hit = r.hasResult && picks.some((x) => paid.includes(x.combo));
    const res = r.hasResult && ok ? `<span class="hitmark ${hit ? 'yes' : 'no'}">${hit ? '的中' : '外れ'}</span>` : '';
    rows.push(`<div class="rec ${ok ? '' : 'off'}">
      <div class="rec-h"><b>${esc(rc.title)}</b><small>過去の成績: 的中${pct(rc.hold.hit)}%・回収${pct(rc.hold.roi)}%</small></div>
      <div class="rec-b">${ok ? picks.map((x) => `${comboHtml(x.combo)} <span class="p">${pct(x.p)}%</span>`).join('　') : `<span class="explain">このレースは対象外（本命の確率 ${pct(list[0].p)}% ＜ ${pct(conf)}%）</span>`} ${res}</div>
    </div>`);
  }
  return rows.length ? `<h3>おすすめの買い方<small>過去の比較で選んだ方法</small></h3>${rows.join('')}` : '';
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
  } else {
    out.push('まだ展示前のため、出走表と近況だけで計算しています。展示が終わると自動で直前情報を加えて計算し直します。');
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
    if (sv !== null && sv <= -0.05 && pr.course[i] >= 2) out.push(`${name(i)} は内側の艇より展示STが ${Math.abs(sv).toFixed(2)} 早い（攻めやすい）`);
  }
  const fs = B.filter((b) => b.F > 0);
  if (fs.length) out.push(`F持ち（スタートを控えやすい）: ${fs.map((b) => chip(b.waku)).join(' ')}`);
  return out;
}

function renderScore(rs) {
  const done = rs.filter((r) => r.hasResult && state.preds.get(r.id) && (r.payouts.trifecta || []).length);
  if (!done.length) return '';
  let win = 0, tri5 = 0, ret = 0, cost = 0;
  for (const r of done) {
    const pr = state.preds.get(r.id);
    const h = topHit(r, pr);
    win += h.win ? 1 : 0;
    tri5 += h.tri5 ? 1 : 0;
    const tp = r.payouts.trifecta || [];
    for (const t of pr.trifecta.slice(0, 5)) { cost += 100; const x = tp.find((p) => p.combo === t.combo); if (x) ret += x.amount; }
  }
  const m = state.model || {};
  let caveat = '';
  if (m.trained_until && state.date <= m.trained_until) caveat = 'この日のレースは予想モデルの学習に使われているため、実際より良い成績になります（参考）。ずるの無い成績は下の「毎日の記録」を見てください。';
  else if (!state.snapExact) caveat = 'この日の選手の近況データが無いため、別の日の近況で計算しています（参考）。';
  return `<section class="score"><h2>答え合わせ（${done.length}レース）</h2><div class="grid">
    <div><div class="k">1着の的中</div><div class="v">${pct(win / done.length)}<small>%</small></div></div>
    <div><div class="k">3連単 上位5点の的中</div><div class="v">${pct(tri5 / done.length)}<small>%</small></div></div>
    <div><div class="k">上位5点を100円ずつ</div><div class="v">${pct(ret / cost)}<small>%回収</small></div></div>
  </div>${caveat ? `<p class="explain">${caveat}</p>` : ''}</section>`;
}

// 毎日の記録（その日の朝の時点の近況とモデルで予想し直した、ずるの無い成績）
function renderRecord() {
  const days = (state.report && state.report.days) || [];
  if (!days.length) return '<section class="score"><h2>毎日の記録</h2><p class="explain">毎朝、前日の答え合わせを自動で記録します（明日の朝から表示されます）。</p></section>';
  const recent = days.slice(-14).reverse();
  const tot = days.slice(-30).reduce((a, d) => { a.r += d.races; a.w += d.top1; a.t5 += d.tri5; a.ret += d.tri5Ret; return a; }, { r: 0, w: 0, t5: 0, ret: 0 });
  const rows = recent.map((d) => `<tr><td>${+d.date.slice(5, 7)}/${+d.date.slice(8)}</td><td>${d.races}</td><td>${pct(d.top1 / d.races)}%</td><td>${pct(d.tri5 / d.races)}%</td><td>${pct(d.tri5Ret / (d.races * 500))}%</td></tr>`).join('');
  const recTot = {};
  for (const d of days.slice(-30)) for (const [k, v] of Object.entries(d.rec || {})) { const t = recTot[k] || (recTot[k] = { bets: 0, hits: 0, ret: 0 }); t.bets += v.bets; t.hits += v.hits; t.ret += v.ret; }
  const recRows = Object.entries(recTot).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.bets}</td><td>${pct(v.hits / Math.max(v.bets, 1))}%</td><td>${pct(v.ret / Math.max(v.bets * 100, 1))}%</td></tr>`).join('');
  return `<section class="score"><h2>毎日の記録（直近${Math.min(days.length, 30)}日: 1着 ${pct(tot.w / tot.r, 1)}%、3連単5点 的中 ${pct(tot.t5 / tot.r, 1)}%、回収 ${pct(tot.ret / (tot.r * 500))}%）</h2>
    <table class="rec"><thead><tr><th>日付</th><th>レース</th><th>1着</th><th>3連単5点</th><th>回収</th></tr></thead><tbody>${rows}</tbody></table>
    ${recRows ? `<h3>買い方の実際の成績（直近${Math.min(days.length, 30)}日）</h3><table class="rec"><thead><tr><th>買い方</th><th>点数</th><th>的中</th><th>回収</th></tr></thead><tbody>${recRows}</tbody></table>` : ''}
    <p class="explain">その日の朝の時点のデータと予想モデルで計算し直した成績です（後から分かった結果は使っていません）。</p></section>`;
}

// 予想方法の比較（research.json）
function renderResearch() {
  const R = state.research;
  if (!R) return '<div class="note">比較の結果はまだありません。</div>';
  const pt = (R.prob_table || []).filter((x) => x.method !== 'market');
  const best = R.chosen_prob;
  const probRows = pt.map((x) => `<tr class="${x.method === best ? 'chosen' : ''}"><td class="l">${esc(x.label)}${x.method === best ? '<br><b>← 採用</b>' : ''}</td>
    <td>${pct(x.hold_win_top1, 1)}</td><td>${pct(x.hold_tri_top1, 1)}</td><td>${pct(x.hold_tri_top5, 1)}</td><td>${pct(x.hold_trio_top1, 1)}</td></tr>`).join('');
  const mk = (R.prob_table || []).find((x) => x.method === 'market');
  const recs = (R.recommend || []).map((x) => `<tr><td class="l">${esc(x.title)}<br><small>${esc(x.label)}</small></td><td>${pct(x.dev.roi)}%</td><td>${pct(x.hold.roi)}%<br><small>±${pct(x.hold.se)}</small></td><td>${pct(x.hold.hit)}%</td><td>${(x.hold.bets / Math.max(1, (R.hold_days || 1))).toFixed(0)}</td></tr>`).join('');
  const types = Object.values(R.best_by_type_top || {}).map((x) => `<tr><td class="l">${esc(x.label)}<br><small>${esc(x.method_label || x.method)}</small></td><td>${pct(x.dev.roi)}%</td><td>${pct(x.hold.roi)}%<br><small>±${pct(x.hold.se)}</small></td><td>${pct(x.hold.hit)}%</td><td>${(x.hold.bets / Math.max(1, (R.hold_days || 1))).toFixed(0)}</td></tr>`).join('');
  return `<section class="score research">
    <h2>予想方法の比較（${esc(R.when || '')}）</h2>
    <p class="explain">${esc(R.rule || '')}</p>
    <p class="explain">開発期間: ${esc((R.dev || []).join('・'))}／本番テスト期間: ${esc((R.hold || []).join('・'))}。下の数字は本番テスト期間の成績。</p>
    <h3>確率の出し方</h3>
    <table class="rec-t"><thead><tr><th class="l">方法</th><th>1着</th><th>3連単<br>本命</th><th>3連単<br>上位5点</th><th>3連複<br>本命</th></tr></thead><tbody>${probRows}</tbody></table>
    ${mk ? `<p class="explain">参考: 確定オッズ（締切後にしか分からない）だけで予想すると 1着${pct(mk.hold_win_top1, 1)}%・3連単本命${pct(mk.hold_tri_top1, 1)}%。締切前のオッズが取れたレースは、予想にオッズを混ぜています。</p>` : ''}
    <h3>おすすめの買い方</h3>
    <table class="rec-t"><thead><tr><th class="l">買い方</th><th>開発<br>回収</th><th>本番<br>回収</th><th>的中</th><th>1日の点数<br>(全場)</th></tr></thead><tbody>${recs}</tbody></table>
    <h3>券種ごとの一番良い買い方</h3>
    <table class="rec-t"><thead><tr><th class="l">買い方</th><th>開発<br>回収</th><th>本番<br>回収</th><th>的中</th><th>1日の点数<br>(全場)</th></tr></thead><tbody>${types}</tbody></table>
    ${renderEvResearch()}
    <p class="explain">${(R.n_strategies || 0).toLocaleString()}通りの買い方を比べました。どの買い方も、安定して回収率100%を超えるものは見つかっていません（舟券は売上の約25%が差し引かれるため）。単勝・複勝などの確定オッズを見て選ぶと回収率が200%を超えるように見えますが、締切直前の大口買いでオッズが歪んだレースを後から選んでいるだけで、実際には買えないため除いています。</p>
  </section>`;
}

function renderEvResearch() {
  const E = state.ev;
  if (!E) return '';
  const sw = (E.exhibit_sweep || []).filter((x) => [0, 0.3, 0.5, 0.7, 0.85, 1].includes(x.w));
  const ew = state.model && state.model.exhibit_w;
  const sweep = sw.map((x) => `<tr class="${x.w === ew ? 'chosen' : ''}"><td class="l">${x.w === 0 ? '0（総合）' : x.w}${x.w === ew ? ' ← 展示最重視' : ''}</td><td>${(x.importance['展示航走'] * 100).toFixed(1)}</td><td>${(x.importance['選手の力'] * 100).toFixed(1)}</td><td>${(x.importance['近況'] * 100).toFixed(1)}</td><td>${pct(x.win_top1, 1)}</td><td>${pct(x.tri_top1, 1)}</td><td>${pct(x.tri_top5, 1)}</td></tr>`).join('');
  const row = (x, name) => (x ? `<tr><td class="l">${name}<br><small>${esc(x.label)}${x.alpha < 1 ? `（予想${pct(x.alpha)}%＋オッズ）` : ''}</small></td><td>${pct(x.dev.roi)}%</td><td>${pct(x.hold.roi)}%<br><small>±${pct(x.hold.se)}</small></td><td>${pct(x.hold.hit, 1)}%</td><td>${(x.hold.bets / Math.max(1, (E.hold_days || 120))).toFixed(0)}</td></tr>` : '');
  const ws = Object.keys(E.by_w || {});
  const evRows = ws.map((w) => { const b = E.by_w[w]; const nm = w === '0' ? '総合' : '展示最重視'; return row(b.trifecta, `3連単・${nm}`) + row(b.exacta, `2連単・${nm}`); }).join('');
  return `<h3>展示航走の重さ<small>展開モデル＋（枠＋展示航走のモデル）を混ぜる割合</small></h3>
    ${sweep ? `<table class="rec-t"><thead><tr><th class="l">展示の割合</th><th>効き<br>展示</th><th>効き<br>選手</th><th>効き<br>近況</th><th>1着</th><th>3連単<br>本命</th><th>3連単<br>上位5</th></tr></thead><tbody>${sweep}</tbody></table>
    <p class="explain">効き＝その材料をレース内で平均にならしたときに、3連単の予想がどれだけ悪くなるか（大きいほど予想を左右している）。枠（コース）はどの割合でも残しています。展示の割合を0.85にすると、枠以外では展示航走がいちばん効く材料になりますが、的中率は総合より下がります。</p>` : ''}
    <h3>期待値の買い方（3連単・2連単）</h3>
    <table class="rec-t"><thead><tr><th class="l">買い方</th><th>開発<br>回収</th><th>本番<br>回収</th><th>的中</th><th>1日の点数<br>(全場)</th></tr></thead><tbody>${evRows}</tbody></table>
    <p class="explain">開発期間（${esc((E.dev || []).join('・'))}）で一番良かった決め方を、本番テスト期間（${esc((E.hold || []).join('・'))}）で確かめた結果。オッズは確定オッズで計算しているため、締切前のオッズで買う実際の成績はこれより少しぶれます。</p>`;
}

function render() {
  renderStatus();
  const venues = [...new Map(state.races.map((r) => [r.jcd, r.venue])).entries()].sort((a, b) => a[0] - b[0]);
  $('#venues').innerHTML = state.view === 'research' ? '' : [[0, 'すべて'], ...venues].map(([j, v]) => `<button type="button" data-venue="${j}" aria-pressed="${state.venue === j}">${esc(v)}</button>`).join('');
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
  document.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode)));
  $('#modeRow').hidden = state.view === 'research' || !(state.model && state.model.tenkai);
  const list = $('#list');
  if (state.view === 'research') { list.innerHTML = renderResearch(); renderFooter(); return; }
  if (state.error && !state.races.length) { list.innerHTML = `<div class="note bad">${esc(state.error)}</div>`; return; }
  if (!state.races.length) { list.innerHTML = state.loading ? '' : '<div class="note">この日のレースはありません。</div>'; return; }
  const rs = filtered();
  let html = state.error ? `<div class="note bad">${esc(state.error)}</div>` : '';
  if (state.view === 'done') html += renderScore(rs) + renderRecord();
  if (!rs.length) html += `<div class="note">${state.view === 'upcoming' ? 'これから締切のレースはありません。「答え合わせ」で結果を見られます。' : 'まだ終わったレースはありません。'}</div>`;
  html += rs.map(raceRow).join('');
  list.innerHTML = html;
  renderFooter();
}

function renderFooter() {
  const m = state.model;
  if (!m) return;
  const v = (m.valid && m.valid.full) || {};
  const ev = m.evaluation || {};
  const lines = [];
  const n = (v.n_train_races || 0) + (v.n_valid_races || 0);
  lines.push(`予想モデル: ${m.trained_until || '-'} までの約${n ? (n / 10000).toFixed(1) : '-'}万レースで学習。${m.tenkai ? '展開モデル（誰がどの決まり手で勝つか→その展開で2着→3着を、選手の組み合わせから予想）に、枠と展示航走だけのモデルを混ぜています。' : ''}（スタート展示・展示タイム・チルト・部品交換・気象・選手とモーターの近況を使用）`);
  const evs = (m.evaluation_compare && m.evaluation_compare.tenkai_w0) || null;
  const one = (e) => `1着の的中 ${pct(e.top1, 1)}%、3連単 本命の的中 ${pct(e.trifecta['1'].hit, 1)}%、上位5点の的中 ${pct(e.trifecta['5'].hit, 1)}%（上位5点を買い続けた回収率 ${pct(e.trifecta['5'].roi)}%）`;
  if (ev.top1) {
    lines.push(`学習に使っていないレースでの検証（${ev.period ? `${ev.period[0]}〜${ev.period[1]}` : ''}、${ev.races.toLocaleString()}レース、ページと同じ計算）:`);
    if (m.tenkai && evs) {
      lines.push(`展示最重視: ${one(ev)}。`);
      lines.push(`総合（的中率重視）: ${one(evs)}。展示を最重視すると的中率は下がります。`);
    } else lines.push(`${one(ev)}。`);
  }
  lines.push('データ: Boatrace Open API（非公式・約10分おき更新）と、公式サイトから自動で取った直前情報。予想は参考情報です。舟券の購入は20歳以上、余裕のある範囲で。');
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
  const vb = e.target.closest('[data-venue]');
  if (vb) { state.venue = +vb.dataset.venue; render(); return; }
  const mb = e.target.closest('[data-mode]');
  if (mb) {
    state.mode = mb.dataset.mode; savePref('mode', state.mode);
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
