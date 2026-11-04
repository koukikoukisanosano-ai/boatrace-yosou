// 予想ページ: データを読み込んで、スマホの中で予想を計算して表示する
import { normalizeDay, prepareModel, predictRace, Snapshot, previewState, FEATURE_INDEX, RANK_LABEL } from './core.js';

const OPENAPI = 'https://boatraceopenapi.github.io/api/v1';
const REFRESH_MS = 60 * 1000;
const WIND = ['', '北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東', '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西', '無風'];
const WEATHER = { 1: '晴', 2: '曇り', 3: '雨', 4: '雪', 5: '台風', 6: '霧' };
const MARKS = ['◎', '○', '▲', '△'];

const state = {
  date: jstToday(), model: null, snap: null, snapNote: '', races: [], preds: new Map(),
  openApiAt: null, liveAt: null, liveCount: 0, view: 'upcoming', venue: 0, open: new Set(),
  error: '', liveError: false, loading: false, loadedAt: null,
};

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
  } catch (e) {
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
  if (!live || !live.races) return;
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
    if (full || !state.snap) await loadSnapshot(state.date);
    const [races, live] = await Promise.all([loadDay(state.date), loadLive(state.date)]);
    mergeLive(races, live);
    state.races = races;
    state.preds.clear();
    for (const r of races) {
      try { const p = predictRace(r, state.model, state.snap); if (p) state.preds.set(r.id, p); } catch (e) { console.error(r.id, e); }
    }
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
  let cum = 0;
  const bets = top.map((t, k) => {
    cum += t.p;
    const hit = tp.includes(t.combo);
    return `<div class="${hit ? 'hit' : ''}">${comboHtml(t.combo)}</div><div class="p ${hit ? 'hit' : ''}">${pct(t.p, 1)}%</div><div class="o ${hit ? 'hit' : ''}">目安 <b>${(1 / t.p).toFixed(1)}</b>倍</div>`;
  }).join('');
  const cum5 = pr.trifecta.slice(0, 5).reduce((a, t) => a + t.p, 0);
  const minor = (list, n, label) => `<div><b>${label}</b> ${list.slice(0, n).map((x) => `${comboHtml(x.combo)}<span class="p">${pct(x.p)}%</span>`).join('　')}</div>`;

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
    <h3>3連単の確率<small>上位5点で ${pct(cum5)}%</small></h3>
    <div class="bets">${bets}</div>
    <p class="explain">目安＝確率から計算したつり合いの倍率。実際のオッズがこれより高ければ、予想どおりなら得な買い目です。</p>
    <h3>ほかの券種</h3>
    <div class="minor">${minor(pr.exacta, 3, '2連単')}${minor(pr.trio, 3, '3連複')}${minor(pr.quinella, 2, '2連複')}</div>
    <h3>予想の根拠</h3>
    <ul class="reasons">${reasons(r, pr).map((x) => `<li>${x}</li>`).join('')}</ul>
    ${result}`;
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
  return `<section class="score"><h2>答え合わせ（${done.length}レース）</h2><div class="grid">
    <div><div class="k">1着の的中</div><div class="v">${pct(win / done.length)}<small>%</small></div></div>
    <div><div class="k">3連単 上位5点の的中</div><div class="v">${pct(tri5 / done.length)}<small>%</small></div></div>
    <div><div class="k">上位5点を100円ずつ</div><div class="v">${pct(ret / cost)}<small>%回収</small></div></div>
  </div></section>`;
}

function render() {
  renderStatus();
  const venues = [...new Map(state.races.map((r) => [r.jcd, r.venue])).entries()].sort((a, b) => a[0] - b[0]);
  $('#venues').innerHTML = [[0, 'すべて'], ...venues].map(([j, v]) => `<button type="button" data-venue="${j}" aria-pressed="${state.venue === j}">${esc(v)}</button>`).join('');
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
  const list = $('#list');
  if (state.error && !state.races.length) { list.innerHTML = `<div class="note bad">${esc(state.error)}</div>`; return; }
  if (!state.races.length) { list.innerHTML = state.loading ? '' : '<div class="note">この日のレースはありません。</div>'; return; }
  const rs = filtered();
  let html = state.error ? `<div class="note bad">${esc(state.error)}</div>` : '';
  if (state.view === 'done') html += renderScore(rs);
  if (!rs.length) html += `<div class="note">${state.view === 'upcoming' ? 'これから締切のレースはありません。「終了・答え合わせ」で結果を見られます。' : 'まだ終わったレースはありません。'}</div>`;
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
  lines.push(`予想モデル: ${m.trained_until || '-'} までの約${n ? (n / 10000).toFixed(1) : '-'}万レースで学習（スタート展示・展示タイム・チルト・部品交換・気象・選手とモーターの近況を使用）。`);
  if (ev.top1) lines.push(`学習に使っていないレースでの検証（${ev.period ? `${ev.period[0]}〜${ev.period[1]}` : ''}、${ev.races.toLocaleString()}レース）: 1着の的中 ${pct(ev.top1, 1)}%、3連単 上位5点の的中 ${pct(ev.trifecta['5'].hit, 1)}%、上位5点を買い続けた場合の回収率 ${pct(ev.trifecta['5'].roi)}%。`);
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
