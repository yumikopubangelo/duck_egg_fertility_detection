/* ── batch.js ── multi-image prediction ── */

const dropZone   = document.getElementById('batch-drop-zone');
const fileInput  = document.getElementById('batch-file-input');
const chips      = document.getElementById('file-chips');
const btnRun     = document.getElementById('btn-run-batch');
const btnClear   = document.getElementById('btn-clear');
const fileCount  = document.getElementById('file-count');

let selectedFiles = [];
let allResults    = [];

/* ── Drag & drop ─────────────────────────────────────────── */
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => addFiles([...fileInput.files]));

function addFiles(files) {
  const imgs = files.filter(f => f.type.startsWith('image/'));
  imgs.forEach(f => {
    if (!selectedFiles.find(x => x.name === f.name && x.size === f.size)) {
      selectedFiles.push(f);
    }
  });
  renderChips();
}

function renderChips() {
  chips.classList.toggle('d-none', selectedFiles.length === 0);
  chips.innerHTML = selectedFiles.map((f, i) =>
    '<span class="badge bg-light text-dark border d-flex align-items-center gap-1" style="font-size:.78rem;padding:4px 8px">'
    + f.name.length > 20 ? f.name.substring(0, 18) + '…' : f.name
    + '<button type="button" class="btn-close btn-close-sm ms-1" style="font-size:.55rem" data-idx="' + i + '"></button>'
    + '</span>'
  ).join('');
  chips.querySelectorAll('.btn-close').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      selectedFiles.splice(parseInt(btn.dataset.idx), 1);
      renderChips();
    });
  });
  const n = selectedFiles.length;
  fileCount.textContent = n > 0 ? n + ' gambar dipilih' : '';
  btnRun.disabled   = n === 0;
  btnClear.disabled = n === 0;
}

btnClear.addEventListener('click', () => {
  selectedFiles = [];
  fileInput.value = '';
  renderChips();
  hideResults();
});

/* ── Run batch ───────────────────────────────────────────── */
btnRun.addEventListener('click', async () => {
  if (selectedFiles.length === 0) return;

  btnRun.disabled  = true;
  btnRun.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memproses…';
  hideResults();
  showProgress(0, selectedFiles.length);
  allResults = [];

  for (let i = 0; i < selectedFiles.length; i++) {
    const file = selectedFiles[i];
    updateProgress(i, selectedFiles.length, file.name);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      const d   = await res.json();
      if (d.error) throw new Error(d.error);
      allResults.push({ file, data: d, ok: true });
    } catch (err) {
      allResults.push({ file, data: null, ok: false, error: err.message });
    }
  }

  updateProgress(selectedFiles.length, selectedFiles.length, 'Selesai');
  btnRun.disabled  = false;
  btnRun.innerHTML = '<i class="bi bi-search me-2"></i>Analisis Semua';

  renderResults();
});

/* ── Progress ────────────────────────────────────────────── */
function showProgress(current, total) {
  document.getElementById('progress-section').classList.remove('d-none');
  updateProgress(current, total, '');
}
function updateProgress(current, total, label) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  document.getElementById('progress-bar').style.width  = pct + '%';
  document.getElementById('progress-pct').textContent  = pct + '%';
  document.getElementById('progress-label').textContent =
    current < total ? ('Memproses ' + (current + 1) + '/' + total + ': ' + label) : 'Selesai';
}

/* ── Results ─────────────────────────────────────────────── */
function renderResults() {
  const ok      = allResults.filter(r => r.ok);
  const fertile = ok.filter(r => r.data.prediction === 'fertile');
  const avgConf = ok.length > 0
    ? (ok.reduce((s, r) => s + (r.data.confidence || 0), 0) / ok.length * 100).toFixed(1)
    : '—';

  document.getElementById('summary-section').classList.remove('d-none');
  document.getElementById('results-section').classList.remove('d-none');
  document.getElementById('s-total').textContent     = allResults.length;
  document.getElementById('s-fertile').textContent   = fertile.length;
  document.getElementById('s-infertile').textContent = ok.length - fertile.length;
  document.getElementById('s-conf').textContent      = avgConf !== '—' ? avgConf + '%' : '—';

  const tbody = document.getElementById('results-tbody');
  tbody.innerHTML = allResults.map((r, idx) => {
    if (!r.ok) {
      return '<tr class="result-row">'
        + '<td class="ps-3"></td>'
        + '<td class="text-truncate" style="max-width:160px">' + r.file.name + '</td>'
        + '<td class="text-center" colspan="4"><span class="text-muted small">—</span></td>'
        + '<td class="text-center pe-3"><span class="badge bg-danger">Error</span></td>'
        + '</tr>';
    }
    const d          = r.data;
    const isFertile  = d.prediction === 'fertile';
    const confPct    = ((d.confidence || 0) * 100).toFixed(1);
    const purityPct  = ((d.cluster_purity || 0) * 100).toFixed(1);
    const colorClass = isFertile ? 'success' : 'danger';
    const labelID    = isFertile ? 'SUBUR' : 'TIDAK SUBUR';

    return '<tr class="result-row" id="row-' + idx + '">'
      + '<td class="ps-3 py-2"><img src="" alt="" class="thumb" id="thumb-' + idx + '"></td>'
      + '<td class="py-2" style="max-width:180px"><div class="text-truncate small fw-semibold">' + r.file.name + '</div></td>'
      + '<td class="text-center py-2"><span class="badge bg-' + colorClass + '">' + labelID + '</span></td>'
      + '<td class="text-center py-2"><span class="fw-semibold text-' + colorClass + '">' + confPct + '%</span></td>'
      + '<td class="text-center py-2">#' + d.cluster_id + '</td>'
      + '<td class="text-center py-2">' + purityPct + '%</td>'
      + '<td class="text-center pe-3 py-2"><span class="badge bg-success-subtle text-success border border-success-subtle">OK</span></td>'
      + '</tr>';
  }).join('');

  /* Load thumbnails */
  allResults.forEach((r, idx) => {
    if (!r.ok) return;
    const img = document.getElementById('thumb-' + idx);
    if (!img) return;
    const fr = new FileReader();
    fr.onload = e => { img.src = e.target.result; };
    fr.readAsDataURL(r.file);
  });
}

function hideResults() {
  document.getElementById('summary-section').classList.add('d-none');
  document.getElementById('results-section').classList.add('d-none');
  document.getElementById('progress-section').classList.add('d-none');
}

/* ── Export CSV ──────────────────────────────────────────── */
document.getElementById('btn-export').addEventListener('click', () => {
  if (allResults.length === 0) return;

  const header = ['filename', 'prediction', 'label_id', 'confidence', 'cluster_id',
                  'cluster_purity', 'cluster_probability', 'score_fertile',
                  'score_infertile', 'feature_count', 'timestamp'];

  const rows = allResults.map(r => {
    if (!r.ok) return [r.file.name, 'ERROR', '', '', '', '', '', '', '', '', ''];
    const d = r.data;
    const sc = d.label_scores || {};
    return [
      d.original_filename || r.file.name,
      d.prediction,
      d.label_id ?? '',
      (d.confidence || 0).toFixed(4),
      d.cluster_id ?? '',
      (d.cluster_purity || 0).toFixed(4),
      (d.cluster_probability || 0).toFixed(4),
      sc.fertile   != null ? sc.fertile.toFixed(4)   : '',
      sc.infertile != null ? sc.infertile.toFixed(4) : '',
      d.feature_count ?? '',
      d.timestamp ? new Date(d.timestamp).toLocaleString('id-ID') : '',
    ];
  });

  const csv = [header, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'batch_prediksi_' + new Date().toISOString().slice(0,10) + '.csv';
  a.click();
  URL.revokeObjectURL(url);
});
