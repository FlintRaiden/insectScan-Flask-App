/**
 * InsectScan — app.js
 * Handles: drag-drop, file preview, API call, result rendering
 */

// ─── Constants ──────────────────────────────────────────────────────────────
const CLASS_COLORS = {
  Butterfly:   '#8B5CF6',
  Dragonfly:   '#06B6D4',
  Grasshopper: '#22C55E',
  Ladybird:    '#EF4444',
  Mosquito:    '#F59E0B',
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const selectBtn     = document.getElementById('selectBtn');
const dropContent   = document.getElementById('dropContent');
const previewContent= document.getElementById('previewContent');
const previewImg    = document.getElementById('previewImg');
const previewName   = document.getElementById('previewName');
const previewSize   = document.getElementById('previewSize');
const classifyBtn   = document.getElementById('classifyBtn');
const resetBtn      = document.getElementById('resetBtn');
const resultCard    = document.getElementById('resultCard');
const errorCard     = document.getElementById('errorCard');
const errorMsg      = document.getElementById('errorMsg');
const errorResetBtn = document.getElementById('errorResetBtn');

let selectedFile = null;

// ─── Event: Select button ────────────────────────────────────────────────────
selectBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target === dropZone || e.target.closest('.drop-content')) {
    fileInput.click();
  }
});

// ─── Event: File input change ────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
});

// ─── Event: Drag & Drop ──────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

// ─── Event: Reset / re-select ────────────────────────────────────────────────
resetBtn.addEventListener('click', reset);
errorResetBtn.addEventListener('click', reset);

// ─── Event: Classify ─────────────────────────────────────────────────────────
classifyBtn.addEventListener('click', classify);

// ─── File selected ───────────────────────────────────────────────────────────
function handleFileSelected(file) {
  const ALLOWED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!ALLOWED.includes(file.type)) {
    showError('Format file tidak didukung. Gunakan JPG, PNG, WEBP, atau BMP.');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError('Ukuran file terlalu besar. Maksimal 16 MB.');
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewSize.textContent = formatSize(file.size);

    dropContent.style.display  = 'none';
    previewContent.style.display = 'flex';
    resultCard.style.display = 'none';
    errorCard.style.display  = 'none';
  };
  reader.readAsDataURL(file);
}

// ─── Classify (API call) ─────────────────────────────────────────────────────
async function classify() {
  if (!selectedFile) return;

  setLoading(true);
  resultCard.style.display = 'none';
  errorCard.style.display  = 'none';

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();

    if (!data.success) {
      showError(data.error || 'Terjadi kesalahan yang tidak diketahui.');
      return;
    }

    renderResult(data);

  } catch (err) {
    showError('Tidak dapat terhubung ke server. Pastikan Flask berjalan.');
  } finally {
    setLoading(false);
  }
}

// ─── Render result ───────────────────────────────────────────────────────────
function renderResult(data) {
  const { predicted_class, confidence, class_probabilities, class_info, image_url } = data;
  const color  = CLASS_COLORS[predicted_class] || '#4ade80';
  const icon   = class_info.ikon  || '🐛';
  const nameId = class_info.nama_id || predicted_class;
  const desc   = class_info.deskripsi || '';

  // Sort probabilities descending
  const sorted = Object.entries(class_probabilities)
    .sort(([,a],[,b]) => b - a);

  const probRowsHTML = sorted.map(([cls, pct]) => {
    const isWinner = cls === predicted_class;
    const clrBar   = isWinner ? '' : `style="background:${CLASS_COLORS[cls]}66"`;
    return `
      <div class="prob-row${isWinner ? ' is-winner' : ''}">
        <div class="prob-row-header">
          <span class="prob-class-name">
            <span>${getIcon(cls)}</span> ${cls}
          </span>
          <span class="prob-pct">${pct.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" data-pct="${pct}" ${clrBar}></div>
        </div>
      </div>`;
  }).join('');

  resultCard.innerHTML = `
    <div class="result-header">
      <div class="result-img-wrap">
        <img src="${image_url}" alt="Uploaded insect" />
      </div>
      <div class="result-verdict">
        <div class="result-predicted-label">Terdeteksi sebagai</div>
        <div class="result-predicted-class" style="color:${color}">
          <span>${icon}</span> ${predicted_class}
        </div>
        <div class="result-class-id">${nameId}</div>
        <div class="result-confidence-badge">Keyakinan: ${confidence.toFixed(1)}%</div>
      </div>
    </div>
    <div class="result-body">
      ${desc ? `<div class="result-desc">${icon} <strong>${nameId}</strong> — ${desc}</div>` : ''}
      <div class="prob-title">Distribusi Probabilitas Kelas</div>
      <div class="prob-list">${probRowsHTML}</div>
    </div>
    <div class="result-footer">
      <button class="btn-reset" id="classifyAnotherBtn">↩ Klasifikasikan Gambar Lain</button>
    </div>`;

  resultCard.style.display = 'block';

  // Scroll ke result
  setTimeout(() => resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);

  // Animate progress bars
  requestAnimationFrame(() => {
    resultCard.querySelectorAll('.prob-bar-fill').forEach((bar) => {
      const pct = parseFloat(bar.dataset.pct);
      setTimeout(() => { bar.style.width = `${pct}%`; }, 100);
    });
  });

  // Re-classify button
  document.getElementById('classifyAnotherBtn').addEventListener('click', reset);
}

// ─── Show error ───────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.style.display = 'flex';
  resultCard.style.display = 'none';
  setTimeout(() => errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);
}

// ─── Reset ────────────────────────────────────────────────────────────────────
function reset() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  dropContent.style.display    = 'flex';
  previewContent.style.display = 'none';
  resultCard.style.display     = 'none';
  errorCard.style.display      = 'none';
}

// ─── Loading state ───────────────────────────────────────────────────────────
function setLoading(loading) {
  classifyBtn.disabled = loading;
  classifyBtn.querySelector('.btn-text').style.display = loading ? 'none' : 'inline';
  const spinner = classifyBtn.querySelector('.btn-spinner');
  spinner.style.display = loading ? 'inline' : 'none';
  if (loading) spinner.classList.add('spinning');
  else         spinner.classList.remove('spinning');
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function formatSize(bytes) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1024*1024)  return `${(bytes/1024).toFixed(1)} KB`;
  return `${(bytes/(1024*1024)).toFixed(1)} MB`;
}

function getIcon(cls) {
  const icons = {
    Butterfly:   '🦋',
    Dragonfly:   '🪲',
    Grasshopper: '🦗',
    Ladybird:    '🐞',
    Mosquito:    '🦟',
  };
  return icons[cls] || '🐛';
}
