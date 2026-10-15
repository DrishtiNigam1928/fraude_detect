const $ = (id) => document.getElementById(id);

/* ---------- Presets (pre-transaction fields only — no post-txn balances) ---------- */
const presets = {
  fraud1: { type: 'TRANSFER', amount: 181,    hour: 3,  oldOrig: 181,    oldDest: 0 },
  fraud2: { type: 'CASH_OUT', amount: 274000, hour: 2,  oldOrig: 274000, oldDest: 50000 },
  legit1: { type: 'PAYMENT',  amount: 9839,   hour: 14, oldOrig: 170136, oldDest: 0 },
  legit2: { type: 'TRANSFER', amount: 5000,   hour: 11, oldOrig: 80000,  oldDest: 20000 },
};

function loadPreset(key) {
  const p = presets[key];
  $('f-type').value     = p.type;
  $('f-amount').value   = p.amount;
  $('f-hour').value     = p.hour;
  $('f-old-orig').value = p.oldOrig;
  $('f-old-dest').value = p.oldDest;
  updateFlags();
  $('result-panel').classList.remove('visible');
  hideError();
}

function getVals() {
  return {
    type:    $('f-type').value,
    amount:  parseFloat($('f-amount').value)   || 0,
    hour:    parseInt($('f-hour').value, 10)   || 0,
    oldOrig: parseFloat($('f-old-orig').value) || 0,
    oldDest: parseFloat($('f-old-dest').value) || 0,
  };
}

/* ---------- Visual risk-signal panel: quick heuristics, not the model's real logic ---------- */
function calcFlags(v) {
  const highRisk  = v.type === 'TRANSFER' || v.type === 'CASH_OUT';
  const fullMove  = v.oldOrig > 0 && v.amount >= v.oldOrig * 0.98;
  const senderZero = v.oldOrig === 0;
  const destZero  = v.oldDest === 0 && v.amount > 0;
  const isNight   = [0,1,2,3,4,5,22,23].includes(v.hour);
  const largeTx   = v.amount > 100000;
  return [
    { label: 'High-risk type',    active: highRisk,   desc: highRisk   ? `${v.type} — fraud concentrates in TRANSFER/CASH_OUT` : `${v.type} — lower-risk type` },
    { label: 'Near-full sweep',   active: fullMove,    desc: fullMove    ? 'Amount ≈ entire sender balance'                     : 'Partial amount — normal' },
    { label: 'Sender starts at ₹0', active: senderZero, desc: senderZero ? 'Sender balance was already zero'                    : 'Sender had a balance' },
    { label: 'Receiver shows ₹0', active: destZero,   desc: destZero    ? 'Receiver balance is zero going in'                  : 'Receiver had a balance' },
    { label: 'Night transaction', active: isNight,    desc: isNight     ? `Hour ${v.hour} — night window`                       : `Hour ${v.hour} — daytime` },
    { label: 'Large amount',      active: largeTx,    desc: largeTx     ? `₹${v.amount.toLocaleString()} — high value`           : 'Normal transaction size' },
  ];
}

function updateFlags() {
  const v = getVals();
  const flags = calcFlags(v);
  const activeCount = flags.filter((f) => f.active).length;
  $('flag-count').textContent = `${activeCount}/${flags.length} active`;
  $('flags-grid').innerHTML = flags.map((f) => `
    <div class="flag-item ${f.active ? 'active' : 'safe-flag'}">
      <div class="flag-dot ${f.active ? 'danger' : 'ok'}"></div>
      <div>
        <div class="flag-name">${f.label}</div>
        <div class="flag-desc">${f.desc}</div>
      </div>
    </div>`).join('');
}

function showError(msg) {
  const el = $('form-error');
  el.textContent = msg;
  el.hidden = false;
}
function hideError() { $('form-error').hidden = true; }

function showResult(result) {
  const panel = $('result-panel');
  panel.classList.remove('visible');
  void panel.offsetWidth;
  panel.classList.add('visible');

  const isFraud = result.flagged;
  const iconEl = $('verdict-icon');
  iconEl.textContent = isFraud ? '🚨' : '✅';
  iconEl.className = 'verdict-icon-big ' + (isFraud ? 'fraud' : 'legit');

  const labelEl = $('verdict-label');
  labelEl.textContent = isFraud ? 'FLAGGED' : 'LEGITIMATE';
  labelEl.style.color = isFraud ? 'var(--accent)' : 'var(--safe)';

  $('verdict-sub').textContent = isFraud
    ? 'Score is at or above the review line'
    : 'Score is below the review line';

  $('risk-score').textContent = (result.score * 100).toFixed(1);
  $('risk-threshold').textContent = (result.threshold * 100).toFixed(1);

  const expEl = $('explanation');
  if (result.reasons && result.reasons.length) {
    const parts = result.reasons.map((r) => `${r.label} (${r.impact > 0 ? 'raises' : 'lowers'} risk)`);
    expEl.textContent = `Top factors: ${parts.join(' · ')}.`;
  } else {
    expEl.textContent = isFraud
      ? 'Flagged by the model; no single dominant factor.'
      : 'No strong fraud signal in this transaction.';
  }
  expEl.className = 'explanation' + (isFraud ? '' : ' safe-exp');
}

async function analyze() {
  hideError();
  const v = getVals();
  const btn = $('analyze-btn');
  btn.textContent = '⏳ Analyzing...';
  btn.disabled = true;

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: v.type,
        amount: v.amount,
        oldbalanceOrg: v.oldOrig,
        oldbalanceDest: v.oldDest,
        hour: v.hour,
      }),
    });
    const data = await response.json();
    if (!response.ok) { showError(data.error || `Server error: ${response.status}`); return; }
    showResult(data);
  } catch (err) {
    console.error(err);
    showError('Could not reach the server. Make sure app.py is running and you opened http://127.0.0.1:5000 (not a local file).');
  } finally {
    btn.textContent = '⚡ ANALYZE TRANSACTION';
    btn.disabled = false;
  }
}

document.querySelectorAll('[data-preset]').forEach((b) =>
  b.addEventListener('click', () => loadPreset(b.dataset.preset)));
$('analyze-btn').addEventListener('click', analyze);
['f-type', 'f-amount', 'f-hour', 'f-old-orig', 'f-old-dest'].forEach((id) =>
  $(id).addEventListener('input', updateFlags));

/* ---------- Hero stats, model comparison, eval cards — all from /api/metrics ---------- */
const fmtPct = (x, d = 1) => (x * 100).toFixed(d) + '%';
const fmtInt = (x) => Number(x).toLocaleString();

function renderMetrics(m) {
  if (!m.available) {
    $('stat-rows').textContent = '—';
    $('model-rows').innerHTML = '<p class="muted">Run train_eval.py --csv Fraud.csv to see real numbers here.</p>';
    return;
  }
  const { meta, models, bank_rule_baseline: bank } = m;
  const deployed = models[meta.deployed_model];
  const fraudCount = Math.round(meta.rows * meta.overall_fraud_rate);

  $('stat-rows').textContent = fmtInt(meta.rows);
  $('stat-fraud-rate').textContent = fmtPct(meta.overall_fraud_rate, 2);
  $('stat-fraud-count').textContent = `${fmtInt(fraudCount)} fraud cases`;
  $('stat-prauc').textContent = deployed.pr_auc.toFixed(3);
  $('stat-recall').textContent = fmtPct(deployed.recall, 0);
  $('stat-recall-sub').textContent = `precision ${fmtPct(deployed.precision, 0)}`;

  const rows = [
    { name: "Bank's own rule", badge: 'base', recall: bank.recall, color: '#7880a0' },
    { name: 'Logistic regression', badge: 'mid', recall: models['logreg/pre_txn']?.recall, color: '#4cc9f0' },
    { name: 'XGBoost (deployed)', badge: 'best', recall: deployed.recall, color: '#00d68f' },
  ].filter((r) => typeof r.recall === 'number');

  $('model-rows').innerHTML = rows.map((r) => `
    <div class="model-row">
      <div class="model-row-name">${r.name} <span class="model-badge ${r.badge}">${r.badge === 'best' ? 'BEST' : r.badge === 'mid' ? 'BASELINE' : 'EXISTING'}</span></div>
      <div class="model-bar-wrap"><div class="model-bar" style="background:${r.color};width:${(r.recall * 100).toFixed(1)}%"></div></div>
      <div class="model-accuracy" style="color:${r.color}">${fmtPct(r.recall, 0)}</div>
    </div>`).join('');

  $('eval-grid').innerHTML = `
    <div class="eval-card">
      <div class="eval-card-label">PR-AUC</div>
      <div class="eval-card-val" style="color:var(--safe)">${deployed.pr_auc.toFixed(3)}</div>
      <div class="eval-card-sub">XGBoost, deployed</div>
    </div>
    <div class="eval-card">
      <div class="eval-card-label">Recall</div>
      <div class="eval-card-val" style="color:var(--safe)">${fmtPct(deployed.recall, 0)}</div>
      <div class="eval-card-sub">${fmtInt(deployed.fn)} frauds missed</div>
    </div>
    <div class="eval-card">
      <div class="eval-card-label">Precision</div>
      <div class="eval-card-val" style="color:var(--info)">${fmtPct(deployed.precision, 0)}</div>
      <div class="eval-card-sub">${fmtInt(deployed.fp)} false alarms</div>
    </div>
    <div class="eval-card">
      <div class="eval-card-label">Test window</div>
      <div class="eval-card-val" style="color:var(--warn);font-size:20px">${fmtInt(meta.test_rows)}</div>
      <div class="eval-card-sub">rows, never seen in training</div>
    </div>`;

  $('info-box').innerHTML = `
    <div class="info-box-title">ℹ WHY PR-AUC, NOT ROC-AUC?</div>
    <div class="info-box-body">
      Only ${fmtPct(meta.overall_fraud_rate, 2)} of transactions are fraud, so ROC-AUC can look excellent even when
      precision is poor. PR-AUC of ${deployed.pr_auc.toFixed(3)} reflects performance on the rare class directly.
      The review line is chosen assuming a missed fraud costs ${meta.cost_fn / meta.cost_fp}× a false alarm — change
      that ratio in train_eval.py to shift the precision/recall trade-off.
    </div>`;
}

fetch('/api/metrics').then((r) => r.json()).then(renderMetrics).catch(() => renderMetrics({ available: false }));

updateFlags();
