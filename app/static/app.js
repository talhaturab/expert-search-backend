/* ===================================================================
   Expert Search — demo UI client
   Uses safe DOM construction (document.createElement + textContent)
   throughout — no innerHTML on user content.
   =================================================================== */

const API = {
  chat:   '/chat',
  expert: (id) => `/experts/${encodeURIComponent(id)}`,
  health: '/health',
};

const el = (id) => document.getElementById(id);

// ---------- DOM helper ------------------------------------------------

// h('tag', {class: 'x', dataset: {cid: '...'}, title: '...'}, child1, child2, ...)
// Children can be: strings (become text nodes), DOM nodes, or null/undefined/false (skipped).
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k === 'style') Object.assign(el.style, v);
      else if (k === 'onclick') el.addEventListener('click', v);
      else el.setAttribute(k, v);
    }
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    if (typeof c === 'string' || typeof c === 'number') {
      el.appendChild(document.createTextNode(String(c)));
    } else {
      el.appendChild(c);
    }
  }
  return el;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// ---------- Profile cache --------------------------------------------

// Candidate name is embedded inside the profile markdown (first # heading).
// We fetch-once-and-cache because the same candidate can appear in multiple
// lists (RAG, Det, Suggested) and we don't want to re-hit the DB 3× per click.
const profileCache = new Map();

async function fetchProfile(candidateId) {
  if (profileCache.has(candidateId)) return profileCache.get(candidateId);
  const promise = fetch(API.expert(candidateId))
    .then((r) => r.ok ? r.json() : Promise.reject(r))
    .then((j) => j.markdown);
  profileCache.set(candidateId, promise);
  return promise;
}

function nameFromMarkdown(md) {
  const first = md.split('\n').find((l) => l.startsWith('# '));
  return first ? first.replace(/^#\s*/, '').trim() : '';
}

function currentRoleFromMarkdown(md) {
  // Work-experience entries render as:
  //   ### {title} _(current)_
  //   _{company} • {industry} • {country}_
  // We stitch those two lines into a compact "Title @ Company (Industry, Country)" string.
  const lines = md.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^###\s+(.+?)\s*_\(current\)_\s*$/);
    if (m) {
      const title = m[1].trim();
      const subline = (lines[i + 1] || '').trim().replace(/^_|_$/g, '');
      const parts = subline.split('•').map((x) => x.trim()).filter(Boolean);
      if (parts.length >= 2) {
        const [company, ...rest] = parts;
        return `${title} @ ${company} (${rest.join(', ')})`;
      }
      return subline ? `${title} @ ${subline}` : title;
    }
  }
  return '';
}

// ---------- Health check ---------------------------------------------

async function updateStatus() {
  const row = el('statusRow');
  const text = row.querySelector('.status-text');
  try {
    const r = await fetch(API.health);
    const j = await r.json();
    const ok = j.status === 'ok';
    row.className = 'status-row ' + (ok ? 'ok' : 'err');
    text.textContent = ok ? 'Ready' : 'Degraded';
  } catch {
    row.className = 'status-row err';
    text.textContent = 'Offline';
  }
}

// ---------- Submit flow ----------------------------------------------

let activeConversationId = null;

async function runSearch(query) {
  el('results').hidden = false;
  el('loader').hidden = false;
  el('errorCard').hidden = true;
  el('resultsBody').hidden = true;
  animateLoaderStages();

  try {
    const body = { query };
    if (activeConversationId) body.conversation_id = activeConversationId;

    const r = await fetch(API.chat, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const txt = await r.text();
      throw new Error(r.status + ' — ' + txt);
    }
    const data = await r.json();
    activeConversationId = data.conversation_id || null;

    prefetchProfiles(data);
    renderResponse(data);
    el('loader').hidden = true;
    el('resultsBody').hidden = false;
  } catch (err) {
    el('loader').hidden = true;
    el('errorCard').hidden = false;
    el('errorText').textContent = ' ' + (err.message || String(err));
  }
}

let stageTimer;
function animateLoaderStages() {
  const stages = el('loaderStages').querySelectorAll('.stage');
  let i = 0;
  stages.forEach((s, idx) => s.classList.toggle('active', idx === 0));
  clearInterval(stageTimer);
  stageTimer = setInterval(() => {
    i = (i + 1) % stages.length;
    stages.forEach((s, idx) => s.classList.toggle('active', idx === i));
  }, 1800);
}

function prefetchProfiles(data) {
  const ids = new Set();
  for (const list of [data.rag_picks, data.det_picks, data.suggested]) {
    for (const c of list || []) ids.add(c.candidate_id);
  }
  ids.forEach(fetchProfile);
}

// ---------- Rendering ------------------------------------------------

function renderResponse(data) {
  clearInterval(stageTimer);
  renderParsedSpec(data.parsed_spec);
  renderSuggested(data.suggested, data.rag_picks, data.det_picks);
  renderRag(data.rag_picks);
  renderDet(data.det_picks, data.parsed_spec);
  renderReasoning(data.reasoning);
}

function renderParsedSpec(spec) {
  const body = el('specBody');
  clear(body);
  if (!spec) { body.appendChild(emptyNode('No parsed spec.')); return; }

  const grid = h('div', { class: 'spec-grid' });

  const dims = [
    ['Function',  spec.function,  (d) => d.values],
    ['Industry',  spec.industry,  (d) => d.values],
    ['Geography', spec.geography, (d) => d.values, (d) => d.location_type],
    ['Seniority', spec.seniority, (d) => d.levels],
    ['Skills',    spec.skills,    (d) => d.values],
    ['Languages', spec.languages, (d) => d.values, (d) => d.required_proficiency],
  ];

  for (const [label, dim, getValues, getExtra] of dims) {
    if (!dim) continue;
    const values = getValues(dim) || [];
    const extra = getExtra ? getExtra(dim) : null;
    if (values.length === 0 && !extra) continue;

    const weight = typeof dim.weight === 'number' ? dim.weight : 0;

    grid.appendChild(h('div', { class: 'spec-item' },
      h('div', { class: 'spec-label' },
        label,
        dim.required ? h('span', { class: 'spec-required' }, 'required') : null,
      ),
      h('div', { class: 'spec-values' },
        ...values.map((v) => h('span', { class: 'spec-chip' }, String(v))),
        extra ? h('span', { class: 'spec-chip' }, String(extra)) : null,
      ),
      h('div', { class: 'spec-weight-bar' },
        h('div', { class: 'spec-weight-fill',
                   style: { width: (Math.max(0, Math.min(1, weight)) * 100) + '%' } }),
      ),
      h('div', { class: 'spec-weight-label' }, 'weight ' + weight.toFixed(2)),
    ));
  }

  // Meta row
  grid.appendChild(h('div', { class: 'spec-item' },
    h('div', { class: 'spec-label' }, 'Meta'),
    h('div', { class: 'spec-values' },
      spec.min_years_exp ? h('span', { class: 'spec-chip' }, 'min ' + spec.min_years_exp + 'y exp') : null,
      h('span', { class: 'spec-chip' }, 'temporality: ' + (spec.temporality || 'any')),
      spec.is_refinement ? h('span', {
        class: 'spec-chip',
        style: { background: 'var(--accent-soft)', color: 'var(--accent-ink)' },
      }, 'refinement') : null,
    ),
  ));

  body.appendChild(grid);
}

function sourceForId(id, ragPicks, detPicks) {
  const inRag = (ragPicks || []).some((c) => c.candidate_id === id);
  const inDet = (detPicks || []).some((c) => c.candidate_id === id);
  if (inRag && inDet) return { label: 'Both',  cls: 'source-both' };
  if (inRag)          return { label: 'RAG',   cls: 'source-rag' };
  if (inDet)          return { label: 'Det',   cls: 'source-det' };
  return { label: '—', cls: '' };
}

function candNameCell(cid) {
  const name = h('span', { class: 'cand-name', 'data-slot': 'name' }, cid.slice(0, 8) + '…');
  const role = h('div', { class: 'cand-role', 'data-slot': 'role' }, 'Loading…');
  return h('td', {}, name, role);
}

function highlightsNode(hs) {
  if (!hs || hs.length === 0) return null;
  return h('ul', { class: 'highlights' },
    ...hs.map((x) => h('li', { class: 'highlight-pill' }, String(x))),
  );
}

function matchCell(text, hs) {
  const td = h('td', { class: 'match-cell' }, text || '');
  const hn = highlightsNode(hs);
  if (hn) td.appendChild(hn);
  return td;
}

function scoreCell(score0to100) {
  const s = Math.round(score0to100);
  return h('td', { class: 'score-cell' },
    h('div', { class: 'score-wrap' },
      h('div', { class: 'score-bar' },
        h('div', { class: 'score-bar-fill', style: { width: s + '%' } }),
      ),
      h('span', { class: 'score-num' }, String(s)),
    ),
  );
}

function buildTable(headers, bodyRows) {
  const thead = h('thead', {},
    h('tr', {}, ...headers.map(([label, cls]) => {
      const th = h('th', cls ? { class: cls } : {}, label);
      return th;
    })),
  );
  const tbody = h('tbody', {}, ...bodyRows);
  return h('table', { class: 'result-table' }, thead, tbody);
}

function renderSuggested(suggested, rag, det) {
  const body = el('suggestedBody');
  clear(body);
  if (!suggested || suggested.length === 0) {
    body.appendChild(emptyNode('The judge returned no final picks for this query.'));
    return;
  }

  const rows = suggested.map((c) => {
    const src = sourceForId(c.candidate_id, rag, det);
    const tr = h('tr', { class: 'candidate-row', dataset: { cid: c.candidate_id } },
      h('td', { class: 'rank-cell' }, String(c.rank)),
      candNameCell(c.candidate_id),
      h('td', {}, h('span', { class: 'source-tag ' + src.cls }, src.label)),
      matchCell(c.match_explanation, c.highlights),
    );
    hydrateCandidateRow(tr, c.candidate_id);
    return tr;
  });

  const table = buildTable(
    [['#', 'rank-cell'], ['Candidate', null], ['Source', null], ['Why chosen', 'match-cell']],
    rows,
  );
  body.appendChild(table);
}

function renderRag(picks) {
  const body = el('ragBody');
  clear(body);
  if (!picks || picks.length === 0) { body.appendChild(emptyNode('No RAG picks.')); return; }

  const rows = picks.map((c) => {
    const tr = h('tr', { class: 'candidate-row', dataset: { cid: c.candidate_id } },
      h('td', { class: 'rank-cell' }, String(c.rank)),
      candNameCell(c.candidate_id),
      scoreCell(c.score || 0),
      matchCell(c.match_explanation, c.highlights),
    );
    hydrateCandidateRow(tr, c.candidate_id);
    return tr;
  });

  const table = buildTable(
    [['#', 'rank-cell'], ['Candidate', null], ['LLM score', 'score-cell'], ['Match', 'match-cell']],
    rows,
  );
  body.appendChild(table);
}

// Pick the dims to show as pills based on which dims the parser extracted.
function activeDimsFor(spec) {
  const all = ['function', 'industry', 'geography', 'seniority', 'skills', 'languages'];
  if (!spec) return all;
  const active = [];
  if (spec.function?.values?.length)   active.push('function');
  if (spec.industry?.values?.length)   active.push('industry');
  if (spec.geography?.values?.length)  active.push('geography');
  if (spec.seniority?.levels?.length)  active.push('seniority');
  if (spec.skills?.values?.length)     active.push('skills');
  if (spec.languages?.values?.length)  active.push('languages');
  return active.length ? active : all;
}

function dimClassFor(score) {
  if (score == null) return 'na';
  if (score >= 0.7)  return 'ok';
  if (score >= 0.3)  return 'mid';
  return 'low';
}
function dimGlyphFor(score) {
  if (score == null) return '—';
  if (score >= 0.7)  return '✓';
  if (score >= 0.3)  return '~';
  return '✗';
}
function dimLabel(d) {
  return { function: 'Function', industry: 'Industry', geography: 'Geography',
           seniority: 'Seniority', skills: 'Skills', languages: 'Languages' }[d] || d;
}

function renderDet(picks, spec) {
  const body = el('detBody');
  clear(body);
  if (!picks || picks.length === 0) { body.appendChild(emptyNode('No deterministic picks.')); return; }

  const dims = activeDimsFor(spec);

  const rows = picks.map((c) => {
    const total = Math.round((c.score || 0) * 100);
    const per = c.per_dim || {};
    const dimCells = dims.map((d) => {
      const score = per[d];
      const cls = dimClassFor(score);
      const glyph = dimGlyphFor(score);
      const scoreStr = (typeof score === 'number') ? score.toFixed(2) : '—';
      return h('td', { class: 'dim-cell' },
        h('div', { class: 'dim-pill ' + cls, title: d + ': ' + scoreStr }, glyph),
        h('span', { class: 'dim-score' }, scoreStr),
      );
    });
    const tr = h('tr', { class: 'candidate-row', dataset: { cid: c.candidate_id } },
      h('td', { class: 'rank-cell' }, String(c.rank)),
      candNameCell(c.candidate_id),
      ...dimCells,
      scoreCell(total),
    );
    hydrateCandidateRow(tr, c.candidate_id);
    return tr;
  });

  const headers = [
    ['#', 'rank-cell'],
    ['Candidate', null],
    ...dims.map((d) => [dimLabel(d), 'dim-cell']),
    ['Total', 'score-cell'],
  ];
  body.appendChild(buildTable(headers, rows));

  // Per-row match explanations below the table.
  const expList = h('div', { class: 'det-explanations', style: { marginTop: '12px' } },
    ...picks.map((c) => {
      const wrap = h('div', {
        style: {
          padding: '10px 12px',
          borderTop: '1px solid var(--border)',
          fontSize: '12px',
          color: 'var(--ink-3)',
        },
      },
        h('strong', { style: { color: 'var(--ink-2)' } }, '#' + c.rank),
        '  ',
        c.match_explanation || '',
      );
      const hs = highlightsNode(c.highlights);
      if (hs) wrap.appendChild(hs);
      return wrap;
    }),
  );
  body.appendChild(expList);
}

function renderReasoning(text) {
  const card = el('reasoningCard');
  if (!text || !text.trim()) { card.hidden = true; return; }
  card.hidden = false;
  el('reasoningText').textContent = text;
}

function emptyNode(msg) {
  return h('div', { class: 'empty-state' }, msg);
}

// Hydrate a candidate row with name + current role from their profile.
async function hydrateCandidateRow(tr, cid) {
  const nameSlot = tr.querySelector('[data-slot="name"]');
  const roleSlot = tr.querySelector('[data-slot="role"]');
  try {
    const md = await fetchProfile(cid);
    const name = nameFromMarkdown(md) || cid;
    const role = currentRoleFromMarkdown(md);
    clear(nameSlot);
    nameSlot.appendChild(document.createTextNode(name));
    nameSlot.appendChild(document.createTextNode(' '));
    nameSlot.appendChild(h('span', { class: 'cand-name-id' }, cid.slice(0, 8)));
    roleSlot.textContent = role || '—';
  } catch {
    roleSlot.textContent = '(profile unavailable)';
  }
}

// ---------- Drawer ---------------------------------------------------

async function openDrawer(cid) {
  const drawer = el('drawer');
  drawer.setAttribute('aria-hidden', 'false');
  el('drawerTitle').textContent = 'Loading…';
  const body = el('drawerBody');
  clear(body);
  body.appendChild(h('div', { class: 'drawer-loader' }, 'Loading profile…'));

  try {
    const md = await fetchProfile(cid);
    const name = nameFromMarkdown(md) || cid;
    el('drawerTitle').textContent = name;
    clear(body);
    renderMarkdownInto(body, md);
  } catch {
    clear(body);
    const err = h('div', { class: 'empty-state' }, 'Could not load profile for ');
    err.appendChild(h('code', {}, cid));
    err.appendChild(document.createTextNode('.'));
    body.appendChild(err);
  }
}

function closeDrawer() {
  el('drawer').setAttribute('aria-hidden', 'true');
}

// ---------- Markdown (DOM construction, no innerHTML) ----------------

// Handles: headings, bold, italic, code spans, ul, hr, paragraphs, tables.
function renderMarkdownInto(container, md) {
  const lines = md.split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Heading
    const hdr = line.match(/^(#{1,6})\s+(.*)$/);
    if (hdr) {
      const lvl = hdr[1].length;
      const node = document.createElement('h' + lvl);
      appendInline(node, hdr[2]);
      container.appendChild(node);
      i++; continue;
    }

    // HR
    if (/^\s*---+\s*$/.test(line)) {
      container.appendChild(h('hr'));
      i++; continue;
    }

    // Table
    if (/^\|.*\|\s*$/.test(line) && /^\|[\s:\-|]+\|\s*$/.test(lines[i + 1] || '')) {
      const header = splitRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      const tbl = document.createElement('table');
      const thead = document.createElement('thead');
      const hdrTr = document.createElement('tr');
      for (const cell of header) {
        const th = document.createElement('th');
        appendInline(th, cell);
        hdrTr.appendChild(th);
      }
      thead.appendChild(hdrTr);
      tbl.appendChild(thead);
      const tbody = document.createElement('tbody');
      for (const row of rows) {
        const tr = document.createElement('tr');
        for (const cell of row) {
          const td = document.createElement('td');
          appendInline(td, cell);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      tbl.appendChild(tbody);
      container.appendChild(tbl);
      continue;
    }

    // UL
    if (/^[-*]\s+/.test(line)) {
      const ul = document.createElement('ul');
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        const li = document.createElement('li');
        appendInline(li, lines[i].replace(/^[-*]\s+/, ''));
        ul.appendChild(li);
        i++;
      }
      container.appendChild(ul);
      continue;
    }

    // Blank line
    if (line.trim() === '') { i++; continue; }

    // Paragraph
    const para = [line];
    i++;
    while (i < lines.length && lines[i].trim() !== '' && !/^(#{1,6}|[-*]\s+|\|)/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    const p = document.createElement('p');
    appendInline(p, para.join(' '));
    container.appendChild(p);
  }
}

function splitRow(l) {
  return l.replace(/^\||\|\s*$/g, '').split('|').map((c) => c.trim());
}

// Parse **bold**, *italic*, `code` into DOM nodes inside `parent`.
// Text comes in as raw markdown — we tokenize and emit text nodes + tags.
function appendInline(parent, src) {
  let i = 0;
  const len = src.length;
  let buf = '';
  const flushText = () => {
    if (buf) { parent.appendChild(document.createTextNode(buf)); buf = ''; }
  };
  while (i < len) {
    const ch = src[i];
    // bold
    if (ch === '*' && src[i + 1] === '*') {
      const end = src.indexOf('**', i + 2);
      if (end !== -1) {
        flushText();
        const node = document.createElement('strong');
        appendInline(node, src.slice(i + 2, end));
        parent.appendChild(node);
        i = end + 2;
        continue;
      }
    }
    // italic (asterisk)
    if (ch === '*' && src[i + 1] !== '*') {
      const end = src.indexOf('*', i + 1);
      if (end !== -1) {
        flushText();
        const node = document.createElement('em');
        appendInline(node, src.slice(i + 1, end));
        parent.appendChild(node);
        i = end + 1;
        continue;
      }
    }
    // italic (underscore) — require non-whitespace content so we don't eat
    // stray underscores or ones embedded in identifiers like snake_case.
    if (ch === '_') {
      const end = src.indexOf('_', i + 1);
      if (end !== -1 && end > i + 1) {
        const content = src.slice(i + 1, end);
        if (content.trim() === content && content.length > 0) {
          flushText();
          const node = document.createElement('em');
          appendInline(node, content);
          parent.appendChild(node);
          i = end + 1;
          continue;
        }
      }
    }
    // code
    if (ch === '`') {
      const end = src.indexOf('`', i + 1);
      if (end !== -1) {
        flushText();
        const node = document.createElement('code');
        node.textContent = src.slice(i + 1, end);
        parent.appendChild(node);
        i = end + 1;
        continue;
      }
    }
    buf += ch;
    i++;
  }
  flushText();
}

// ---------- Wire up --------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  updateStatus();

  el('searchForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = el('queryInput').value.trim();
    if (!q) return;
    activeConversationId = null;
    runSearch(q);
  });

  document.querySelectorAll('.example-pill').forEach((btn) => {
    btn.addEventListener('click', () => {
      el('queryInput').value = btn.dataset.query;
      el('searchForm').dispatchEvent(new Event('submit'));
    });
  });

  document.addEventListener('click', (e) => {
    const row = e.target.closest('.candidate-row');
    if (row && row.dataset.cid) openDrawer(row.dataset.cid);
  });

  el('drawerScrim').addEventListener('click', closeDrawer);
  el('drawerClose').addEventListener('click', closeDrawer);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer();
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      el('searchForm').dispatchEvent(new Event('submit'));
    }
  });
});
