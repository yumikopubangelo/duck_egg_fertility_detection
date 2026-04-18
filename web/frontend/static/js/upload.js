const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewBox = document.getElementById('preview-box');
const previewImg = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const btnPredict = document.getElementById('btn-predict');
const resultArea = document.getElementById('result-area');

let selectedFile = null;

// Drag & drop handlers
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewBox.classList.remove('d-none');
    dropZone.classList.add('d-none');
    btnPredict.disabled = false;
  };
  reader.readAsDataURL(file);
}

btnPredict.addEventListener('click', () => {
  if (!selectedFile) return;

  btnPredict.disabled = true;
  btnPredict.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memproses...';
  resultArea.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2 text-muted">Menganalisis telur...</p></div>';

  const formData = new FormData();
  formData.append('file', selectedFile);

  fetch('/api/predict', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(d => {
      if (d.error) throw new Error(d.error);

      const isFertile = d.prediction === 'fertile';
      const confPct = ((d.confidence || 0) * 100).toFixed(1);
      const colorClass = isFertile ? 'success' : 'danger';
      const icon = isFertile ? 'check-circle-fill' : 'x-circle-fill';
      const label = isFertile ? 'FERTILE' : 'INFERTILE';

      resultArea.innerHTML = `
        <div class="text-center w-100 px-2">
          <i class="bi bi-${icon} text-${colorClass}" style="font-size:3.5rem"></i>
          <h3 class="fw-bold mt-2 mb-1 text-${colorClass}">${label}</h3>
          <p class="text-muted mb-3">Telur bebek diprediksi <strong>${label.toLowerCase()}</strong></p>

          <div class="mb-3">
            <div class="d-flex justify-content-between small mb-1">
              <span>Tingkat Keyakinan</span>
              <strong>${confPct}%</strong>
            </div>
            <div class="progress" style="height:10px;border-radius:8px">
              <div class="progress-bar bg-${colorClass}" style="width:${confPct}%;border-radius:8px"></div>
            </div>
          </div>

          <div class="text-start bg-light rounded-3 p-3 small">
            <div class="row g-1">
              <div class="col-6 text-muted">File:</div>
              <div class="col-6 fw-semibold text-truncate">${d.original_filename || selectedFile.name}</div>
              <div class="col-6 text-muted">ID Prediksi:</div>
              <div class="col-6 font-monospace small">${(d.id || '-').substring(0,8)}...</div>
              <div class="col-6 text-muted">Waktu:</div>
              <div class="col-6">${d.timestamp ? new Date(d.timestamp).toLocaleString('id-ID') : '-'}</div>
            </div>
          </div>

          <button class="btn btn-outline-secondary mt-3 w-100" onclick="resetUpload()">
            <i class="bi bi-arrow-left me-1"></i>Prediksi Gambar Lain
          </button>
        </div>`;
    })
    .catch(err => {
      resultArea.innerHTML = `
        <div class="text-center text-danger py-3">
          <i class="bi bi-exclamation-circle fs-2 d-block mb-2"></i>
          <p class="mb-0">Gagal melakukan prediksi.<br><small class="text-muted">${err.message}</small></p>
        </div>`;
    })
    .finally(() => {
      btnPredict.disabled = false;
      btnPredict.innerHTML = '<i class="bi bi-search me-2"></i>Prediksi Sekarang';
    });
});

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewBox.classList.add('d-none');
  dropZone.classList.remove('d-none');
  btnPredict.disabled = true;
  resultArea.innerHTML = `
    <div class="text-center text-muted py-4">
      <i class="bi bi-hourglass fs-1 d-block mb-2"></i>
      <p class="mb-0">Hasil akan muncul di sini setelah prediksi</p>
    </div>`;
}
