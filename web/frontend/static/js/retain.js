function startRetrain() {
  const modelName = document.getElementById('model-name').value;
  const trainRatio = parseFloat(document.getElementById('train-ratio').value);
  const valRatio = parseFloat(document.getElementById('val-ratio').value);
  const testRatio = parseFloat(document.getElementById('test-ratio').value);
  const sum = (trainRatio + valRatio + testRatio).toFixed(2);

  if (Math.abs(sum - 1.0) > 0.01) {
    alert(`Total rasio harus 1.0 (sekarang ${sum})`);
    return;
  }

  const btn = document.getElementById('btn-retrain');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memproses...';

  const statusArea = document.getElementById('retrain-status-area');
  statusArea.innerHTML = `
    <div class="text-center py-3">
      <div class="spinner-border text-warning mb-3"></div>
      <p class="fw-semibold mb-1">Training sedang berjalan...</p>
      <p class="text-muted small mb-0">Model: <strong>${modelName}</strong> &bull; Rasio: ${trainRatio}/${valRatio}/${testRatio}</p>
    </div>`;

  fetch('/api/retrain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName, parameters: { train_ratio: trainRatio, val_ratio: valRatio, test_ratio: testRatio } })
  })
    .then(r => r.json())
    .then(d => {
      const isOk = d.status === 'completed';
      const logs = (d.logs || []).map(l => `<li class="small">${l}</li>`).join('');
      statusArea.innerHTML = `
        <div class="d-flex align-items-center gap-2 mb-3">
          <span class="badge bg-${isOk ? 'success' : 'danger'} px-3 py-2">
            <i class="bi bi-${isOk ? 'check-circle' : 'x-circle'} me-1"></i>${isOk ? 'Selesai' : 'Gagal'}
          </span>
          <span class="text-muted small">Job ID: ${(d.id || '-').substring(0,8)}...</span>
        </div>
        ${logs ? `<ul class="mb-0 ps-3">${logs}</ul>` : ''}`;
      loadHistory();
    })
    .catch(() => {
      statusArea.innerHTML = '<div class="alert alert-danger py-2 mb-0">Gagal menjalankan retrain. Pastikan server berjalan.</div>';
    })
    .finally(() => {
      btn.disabled = false;
      btn.innerHTML = '<i class="bi bi-play-circle me-2"></i>Mulai Retrain';
    });
}

function loadHistory() {
  const tbody = document.getElementById('history-tbody');
  tbody.innerHTML = '<tr><td colspan="5" class="text-center py-3 text-muted"><div class="spinner-border spinner-border-sm me-2"></div>Memuat...</td></tr>';

  fetch('/api/retrain/history')
    .then(r => r.json())
    .then(d => {
      if (!d.jobs || d.jobs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center py-3 text-muted">Belum ada riwayat training</td></tr>';
        return;
      }
      tbody.innerHTML = d.jobs.map(j => `
        <tr>
          <td class="ps-3">${j.model_name}</td>
          <td><span class="badge bg-${j.status === 'completed' ? 'success' : j.status === 'running' ? 'warning' : 'danger'}">${j.status}</span></td>
          <td>${j.accuracy ? (j.accuracy * 100).toFixed(1) + '%' : '-'}</td>
          <td>${j.duration || '-'}</td>
          <td class="text-muted small">${j.started_at ? new Date(j.started_at).toLocaleString('id-ID') : '-'}</td>
        </tr>`).join('');
    })
    .catch(() => {
      tbody.innerHTML = '<tr><td colspan="5" class="text-center py-3 text-danger">Gagal memuat riwayat</td></tr>';
    });
}

loadHistory();
