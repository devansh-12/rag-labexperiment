/* ═══════════════════════════════════════════════════════════════════════════
   SPIT RAG Chatbot – main.js
   Handles: config state · API calls · chat rendering · metrics · charts
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  retriever: 'hybrid',
  k:          3,
  chunker:    'C1',
  tau:        0.0,
  embed:      'A',
  availableIndexes: [],
  loading:    false,
  latencyHistory: [],   // [{retrieve, rerank, generate}]
  currentChunks: [],
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const chatMessages   = $('chatMessages');
const chatForm       = $('chatForm');
const questionInput  = $('questionInput');
const sendBtn        = $('sendBtn');
const welcomeState   = $('welcomeState');
const chunksList     = $('chunksList');
const chunkCountBadge= $('chunkCountBadge');
const statusBadge    = $('statusBadge');
const statusText     = $('statusText');
const modePill       = $('modePill');

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initConfig();
  initInputAutoResize();
  initQuickQuestions();
  checkStatus();
  drawHistoryChart([]);
});

// ── Config wiring ─────────────────────────────────────────────────────────────
function initConfig() {
  // Segmented control
  document.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.retriever = btn.dataset.val;
      syncHints();
    });
  });

  // k slider
  $('kSlider').addEventListener('input', e => {
    state.k = +e.target.value;
    $('kDisplay').textContent = state.k;
    syncHints();
  });

  // tau slider
  $('tauSlider').addEventListener('input', e => {
    state.tau = (e.target.value / 100).toFixed(2);
    $('tauDisplay').textContent = state.tau;
  });

  // selects
  $('chunkerSelect').addEventListener('change', e => {
    state.chunker = e.target.value;
    selectAvailableIndex('chunker');
    syncHints();
  });
  $('embedSelect').addEventListener('change', e => {
    state.embed = e.target.value;
    selectAvailableIndex('embed');
  });
  document.querySelectorAll('.eval-btn').forEach(btn => {
    btn.addEventListener('click', () => runEvaluation(btn.dataset.evaluator));
  });

  // clear
  $('clearBtn').addEventListener('click', clearChat);
  $('retrieverControl').addEventListener('click', updateThresholdAvailability);

  syncHints();
  updateThresholdAvailability();
}

function syncHints() {
  $('hintRetriever').textContent = state.retriever;
  $('hintK').textContent         = state.k;
  $('hintChunker').textContent   = state.chunker;
}

// ── Auto-resize textarea & enter key ──────────────────────────────────────────
function initInputAutoResize() {
  questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
  });
  questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuestion();
    }
  });
}

// ── Quick questions ───────────────────────────────────────────────────────────
function initQuickQuestions() {
  document.querySelectorAll('.quick-q').forEach(btn => {
    btn.addEventListener('click', e => {
      e.preventDefault();
      const q = btn.textContent.trim();
      questionInput.value = q;
      submitQuestion(q);
    });
  });
}

// ── Status check ──────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    configureAvailableIndexes(data.available_indexes || []);
    if (data.pipeline_ready) {
      statusText.textContent = 'Pipeline Ready';
      modePill.textContent = 'LIVE';
      modePill.classList.add('live');
    } else {
      statusText.textContent = 'Demo Mode';
    }
    $('evaluationStatus').textContent =
      `Evaluation set: ${data.qa_count || 0} questions. Uses the selected retriever/index; provider quotas may limit large runs.`;
  } catch {
    statusText.textContent = 'Offline';
    statusBadge.style.borderColor = 'rgba(244,63,94,.3)';
  }
}

// ── Send question ─────────────────────────────────────────────────────────────
async function submitQuestion(forcedText) {
  const question = (forcedText || questionInput.value).trim();
  if (!question || state.loading) return;

  hideWelcome();
  appendUserMsg(question);
  questionInput.value = '';
  questionInput.style.height = 'auto';

  const typingId = appendTyping();
  setLoading(true);

  try {
    const payload = {
      question,
      k:           state.k,
      retriever:   state.retriever,
      chunker:     state.chunker,
      embed:       state.embed,
      threshold:   thresholdEnabled() ? parseFloat(state.tau) : 0,
      temperature: 0.0,
    };

    // Determine target URL (relative or fallback to origin)
    const targetUrl = '/api/chat';

    const res = await fetch(targetUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    removeTyping(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendBotMsg(`⚠️ Error: ${err.detail || res.statusText}`, true);
      return;
    }

    const data = await res.json();
    appendBotMsg(formatAnswer(data.answer, data.cited_ids));
    renderChunks(data.chunks, data.mode);
    renderMetrics(data);
    updateModePill(data.mode);
    showToast('✓ Response received', 'success');

  } catch (err) {
    console.error('Fetch error:', err);
    removeTyping(typingId);
    appendBotMsg(`⚠️ Network error: ${err.message || 'Failed to fetch'}. Please verify connection to http://127.0.0.1:8000`, true);
    showToast('Network error', 'error');
  } finally {
    setLoading(false);
  }
}

// ── Form and send button listeners ───────────────────────────────────────────
chatForm.addEventListener('submit', e => {
  e.preventDefault();
  submitQuestion();
  return false;
});

sendBtn.addEventListener('click', e => {
  e.preventDefault();
  submitQuestion();
});

// ── Message renderers ─────────────────────────────────────────────────────────
function hideWelcome() {
  if (welcomeState) welcomeState.style.display = 'none';
}

function appendUserMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `
    <div class="msg-meta">${timestamp()}</div>
    <div class="bubble bubble-user">${escHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollBottom();
}

function appendBotMsg(html, isError = false) {
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.innerHTML = `
    <div class="msg-meta">SPIT RAG · ${timestamp()}</div>
    <div class="bubble bubble-bot${isError ? ' error' : ''}">${html}</div>`;
  chatMessages.appendChild(div);
  scrollBottom();
}

function appendTyping() {
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.className = 'msg msg-bot'; div.id = id;
  div.innerHTML = `
    <div class="bubble bubble-bot">
      <div class="typing"><span></span><span></span><span></span></div>
    </div>`;
  chatMessages.appendChild(div);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function formatAnswer(text, citedIds = []) {
  // Convert markdown-ish to HTML
  let html = escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>');

  // Style citation tags [S1], [S2] …
  html = html.replace(/\[S(\d+)\]/g, (_, n) =>
    `<span class="cite-tag" title="Source ${n}" onclick="highlightChunk(${+n-1})">[S${n}]</span>`
  );
  return html;
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Chunk panel ───────────────────────────────────────────────────────────────
function renderChunks(chunks, mode) {
  state.currentChunks = chunks;
  chunkCountBadge.textContent = chunks.length;
  chunksList.innerHTML = '';

  if (!chunks.length) {
    const message = mode === 'threshold_rejected'
      ? 'No chunks met the score threshold. Lower it or set it to 0.'
      : 'The selected retriever found no matching chunks.';
    chunksList.innerHTML = `<div class="empty-panel"><p>${message}</p></div>`;
    return;
  }

  const maxScore = Math.max(...chunks.map(c => c.score), 0.001);

  chunks.forEach((chunk, i) => {
    const scoreNorm = Math.min((chunk.score / maxScore) * 100, 100);
    const scoreColor = scoreNorm > 75 ? '#10b981' : scoreNorm > 50 ? '#f59e0b' : '#f43f5e';

    const card = document.createElement('div');
    card.className = 'chunk-card';
    card.id = `chunk-card-${i}`;
    card.innerHTML = `
      <div class="chunk-header">
        <div class="chunk-rank">S${i+1}</div>
        <span class="chunk-source" title="${escHtml(chunk.source)}">${escHtml(chunk.source)}</span>
        <span class="chunk-section">${escHtml(chunk.section || '')}</span>
      </div>
      <div class="score-row">
        <div class="score-track">
          <div class="score-fill" style="width:${scoreNorm.toFixed(1)}%;background:${scoreColor}"></div>
        </div>
        <span class="score-label" style="color:${scoreColor}">${chunk.score.toFixed(3)}</span>
      </div>
      <div class="chunk-text-preview" id="chunk-text-${i}">${escHtml(chunk.text)}</div>`;

    card.addEventListener('click', () => {
      const textEl = $(`chunk-text-${i}`);
      textEl.classList.toggle('expanded');
    });

    chunksList.appendChild(card);

    // Animate score bar after paint
    requestAnimationFrame(() => {
      card.querySelector('.score-fill').style.width = scoreNorm.toFixed(1) + '%';
    });
  });
}

function configureAvailableIndexes(indexes) {
  if (!indexes.length) return;
  state.availableIndexes = indexes;
  const available = (chunker, embed) => indexes.some(
    item => item.chunker === chunker && item.embed === embed
  );
  for (const option of $('chunkerSelect').options) {
    option.disabled = !available(option.value, state.embed);
  }
  for (const option of $('embedSelect').options) {
    option.disabled = !available(state.chunker, option.value);
  }
  if (!available(state.chunker, state.embed)) {
    const first = indexes[0];
    state.chunker = first.chunker;
    state.embed = first.embed;
    $('chunkerSelect').value = first.chunker;
    $('embedSelect').value = first.embed;
  }
  for (const option of $('chunkerSelect').options) {
    option.disabled = !available(option.value, state.embed);
  }
  for (const option of $('embedSelect').options) {
    option.disabled = !available(state.chunker, option.value);
  }
  syncHints();
}

function selectAvailableIndex(changed) {
  const indexes = state.availableIndexes;
  if (!indexes.length || indexes.some(
    item => item.chunker === state.chunker && item.embed === state.embed
  )) return;
  const match = indexes.find(item => item[changed] === state[changed]);
  const fallback = match || indexes[0];
  state.chunker = fallback.chunker;
  state.embed = fallback.embed;
  $('chunkerSelect').value = fallback.chunker;
  $('embedSelect').value = fallback.embed;
  configureAvailableIndexes(indexes);
}

function thresholdEnabled() {
  return ['dense', 'mmr', 'rerank'].includes(state.retriever);
}

function updateThresholdAvailability() {
  const enabled = thresholdEnabled();
  $('tauSlider').disabled = !enabled;
  $('thresholdHint').textContent = enabled
    ? 'Active for the selected dense or rerank scores.'
    : 'Only applies to dense, MMR, and rerank; inactive for hybrid and BM25.';
}

async function runEvaluation(evaluator) {
  const buttons = [...document.querySelectorAll('.eval-btn')];
  const status = $('evaluationStatus');
  buttons.forEach(button => { button.disabled = true; });
  $('evaluationScores').textContent = '';
  status.textContent = `Starting ${evaluator === 'openevals' ? 'OpenEvals' : 'RAGAS'}…`;
  try {
    const response = await fetch('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        evaluator,
        sample_size: Number($('evaluationSize').value),
        k: state.k,
        retriever: state.retriever,
        chunker: state.chunker,
        embed: state.embed,
        threshold: thresholdEnabled() ? Number(state.tau) : 0,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);

    let job;
    do {
      await new Promise(resolve => setTimeout(resolve, 1500));
      const result = await fetch(`/api/evaluation/${payload.job_id}`);
      job = await result.json();
      status.textContent = job.status === 'running'
        ? `${job.phase}: ${job.completed}/${job.total} questions…`
        : status.textContent;
    } while (job.status === 'running');

    if (job.status !== 'completed') throw new Error(job.error || 'Evaluation failed.');
    renderEvaluationSummary(job.summary, job.output_file);
    status.textContent = `${job.evaluator === 'openevals' ? 'OpenEvals' : 'RAGAS'} complete · ${job.completed} questions`;
  } catch (error) {
    status.textContent = `Evaluation failed: ${error.message}`;
  } finally {
    buttons.forEach(button => { button.disabled = false; });
  }
}

function renderEvaluationSummary(summary, outputFile) {
  const root = $('evaluationScores');
  root.innerHTML = '';
  for (const [key, value] of Object.entries(summary || {})) {
    if (value === null || !Number.isFinite(Number(value))) continue;
    const row = document.createElement('div');
    row.className = 'evaluation-score';
    const label = document.createElement('span');
    label.textContent = key.replaceAll('_', ' ');
    const score = document.createElement('strong');
    score.textContent = Number(value).toFixed(3);
    row.append(label, score);
    root.appendChild(row);
  }
  if (outputFile) {
    const output = document.createElement('div');
    output.className = 'cfg-note';
    output.textContent = `Saved: ${outputFile}`;
    root.appendChild(output);
  }
}

window.highlightChunk = function(index) {
  document.querySelectorAll('.chunk-card').forEach((c,i) => {
    c.classList.toggle('highlighted', i === index);
  });
  const card = $(`chunk-card-${index}`);
  if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

// ── Metrics panel ─────────────────────────────────────────────────────────────
function renderMetrics(data) {
  const total = data.t_retrieve + data.t_rerank + data.t_generate;

  $('mRetrieve').textContent = data.t_retrieve.toFixed(3);
  $('mRerank').textContent   = data.t_rerank.toFixed(3);
  $('mGenerate').textContent = data.t_generate.toFixed(3);
  $('mTotal').textContent    = total.toFixed(3);

  // Latency bars
  const mx = Math.max(total, 0.001);
  $('lbRetrieve').style.width = ((data.t_retrieve / mx) * 100).toFixed(1) + '%';
  $('lbRerank').style.width   = ((data.t_rerank   / mx) * 100).toFixed(1) + '%';
  $('lbGenerate').style.width = ((data.t_generate / mx) * 100).toFixed(1) + '%';

  // History
  state.latencyHistory.push({ retrieve: data.t_retrieve, rerank: data.t_rerank, generate: data.t_generate });
  if (state.latencyHistory.length > 20) state.latencyHistory.shift();
  drawHistoryChart(state.latencyHistory);
}

// ── Sparkline (pure canvas, no deps) ─────────────────────────────────────────
function drawHistoryChart(history) {
  const canvas = $('historyChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 260;
  const H = 60;
  canvas.width = W; canvas.height = H;

  ctx.clearRect(0, 0, W, H);

  if (!history.length) {
    ctx.fillStyle = '#4b4b6b';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No runs yet', W/2, H/2 + 4);
    return;
  }

  const totals = history.map(r => r.retrieve + r.rerank + r.generate);
  const maxT   = Math.max(...totals, 0.1);
  const pad    = 8;
  const xStep  = history.length > 1 ? (W - pad*2) / (history.length - 1) : W - pad*2;

  function drawLine(key, color, totalsArr) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    history.forEach((r, i) => {
      const x = pad + i * xStep;
      const y = H - pad - ((r[key] / maxT) * (H - pad*2));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // dots
    history.forEach((r, i) => {
      const x = pad + i * xStep;
      const y = H - pad - ((r[key] / maxT) * (H - pad*2));
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI*2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  drawLine('retrieve', '#7c3aed', totals);
  drawLine('generate', '#10b981', totals);

  // X-axis ticks
  ctx.fillStyle = '#4b4b6b'; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'center';
  history.forEach((_, i) => {
    ctx.fillText(i+1, pad + i * xStep, H - 1);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(val) {
  state.loading = val;
  sendBtn.disabled = val;
}

function scrollBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function timestamp() {
  return new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
}

function clearChat() {
  chatMessages.innerHTML = '';
  chatMessages.appendChild(welcomeState);
  welcomeState.style.display = 'flex';
  chunksList.innerHTML = '<div class="empty-panel"><p>Retrieved chunks will appear here after you send a question.</p></div>';
  chunkCountBadge.textContent = '–';
  ['mRetrieve','mRerank','mGenerate','mTotal'].forEach(id => $(id).textContent = '–');
  ['lbRetrieve','lbRerank','lbGenerate'].forEach(id => $(id).style.width = '0%');
  state.latencyHistory = [];
  drawHistoryChart([]);
}

function updateModePill(mode) {
  modePill.textContent = mode === 'live' ? 'LIVE' : mode === 'demo' ? 'DEMO' : mode.toUpperCase();
  modePill.className = 'mode-pill' + (mode === 'live' ? ' live' : '');
}

function showToast(msg, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}
