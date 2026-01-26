/* =============================================================================
   OCR PDF Web UI - Application Logic
   ============================================================================= */

// =============================================================================
// State
// =============================================================================

const state = {
  inputDir: "",
  outputDir: "",
  items: [],
  selection: new Set(),
  sse: null,
  transitionedToLlm: false,
  
  // Pipeline State
  browsingTarget: "input", // "input" | "output"
  pipeline: {
    outputDir: "",
    llmMode: "disabled", // "disabled" | "after_each" | "after_all"
    llmAvailable: false,
    afterOcr: "llm", // "llm" | "stay"
  },
  
  // Job progress
  jobRunning: false,
  progress: {
    fileIndex: 0,
    totalFiles: 0,
    currentFile: "",
    filePages: 0,
    pagesDone: 0,
    totalPages: 0,
    speed: 0,
    vram: 0,
    eta: "",
    jobSpeed: 0,
    jobEta: "",
    stageName: "",
    stageCurrent: 0,
    stageTotal: 0,
    stageSpeed: 0,
    stageEta: "",
  },
  
  // Sorting
  sortColumn: "name",
  sortAsc: true,
  
  // Settings
  maxPages: 500,
  autoScroll: true,
  
  // Theme
  theme: "system", // "light" | "dark" | "system"
  
  // Performance settings (simplified - fixed config, only auto_fallback is configurable)
  settings: {
    autoFallback: true,
  },
};

// =============================================================================
// Utilities
// =============================================================================

function el(id) {
  return document.getElementById(id);
}

function $(selector) {
  return document.querySelector(selector);
}

function $$(selector) {
  return document.querySelectorAll(selector);
}

// =============================================================================
// Theme
// =============================================================================

function initTheme() {
  const saved = localStorage.getItem("ocr-theme");
  if (saved && ["light", "dark", "system"].includes(saved)) {
    state.theme = saved;
  }
  applyTheme();
  
  // Listen for system preference changes
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.theme === "system") {
      applyTheme();
    }
  });
}

function applyTheme() {
  let dark = false;
  if (state.theme === "dark") {
    dark = true;
  } else if (state.theme === "system") {
    dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

function toggleTheme() {
  const cycle = { light: "dark", dark: "system", system: "light" };
  state.theme = cycle[state.theme];
  localStorage.setItem("ocr-theme", state.theme);
  applyTheme();
  showToast(`Theme: ${state.theme}`, "info");
}

// =============================================================================
// Pipeline Settings
// =============================================================================

function loadPipelineSettings() {
  const saved = localStorage.getItem("ocr-pipeline");
  if (saved) {
    try {
      const p = JSON.parse(saved);
      state.pipeline.outputDir = p.outputDir || "";
      state.pipeline.llmMode = p.llmMode || "disabled";
      state.pipeline.afterOcr = p.afterOcr || "llm";
    } catch {}
  }
  
  // Update UI
  const outEl = el("outputPath");
  if (outEl) outEl.value = state.pipeline.outputDir;
  
  const radios = document.getElementsByName("llmMode");
  for (const r of radios) {
    if (r.value === state.pipeline.llmMode) r.checked = true;
  }

  const afterRadios = document.getElementsByName("afterOcr");
  for (const r of afterRadios) {
    if (r.value === state.pipeline.afterOcr) r.checked = true;
  }
}

function savePipelineSettings() {
  state.pipeline.outputDir = el("outputPath").value.trim();
  
  const radio = document.querySelector('input[name="llmMode"]:checked');
  state.pipeline.llmMode = radio ? radio.value : "disabled";

  const afterRadio = document.querySelector('input[name="afterOcr"]:checked');
  state.pipeline.afterOcr = afterRadio ? afterRadio.value : "llm";
  
  localStorage.setItem("ocr-pipeline", JSON.stringify(state.pipeline));
}

async function triggerLlmPolish(files) {
  if (state.pipeline.llmMode === "disabled") return null;
  if (!state.pipeline.llmAvailable) {
    appendLog("[ui] Skipping auto-polish: LLM offline");
    return null;
  }
  
  // For "after_all" mode, we need to find all files that were just processed
  // These are in the output directory and marked as done but not polished
  let filesToPolish = files;
  
  if (state.pipeline.llmMode === "after_all" && files.length === 0) {
    // Get done files from the output directory
    const outputDir = el("outputPath").value.trim() || state.outputDir;
    if (outputDir) {
      try {
        const res = await fetch(`/api/files/list?dir=${encodeURIComponent(outputDir)}`);
        const data = await res.json();
        if (data.files) {
          // Get all done files that are not polished
          filesToPolish = data.files
            .filter(f => f.done && !f.polished)
            .map(f => f.path);
        }
      } catch (err) {
        appendLog(`[ui] Failed to get files for polish: ${err.message}`);
        return null;
      }
    }
  } else if (state.pipeline.llmMode === "after_each") {
    // For after_each, we need full paths
    const outputDir = el("outputPath").value.trim() || state.outputDir;
    filesToPolish = files.map(f => {
      // If it's just a filename, construct full path
      if (!f.includes("/") && !f.includes("\\")) {
        const mdName = f.replace(/\.pdf$/i, ".md");
        return `${outputDir}/${mdName}`;
      }
      return f;
    });
  }
  
  if (filesToPolish.length === 0) {
    appendLog("[ui] No files to polish");
    return null;
  }
  
  appendLog(`[ui] Starting auto-polish for ${filesToPolish.length} file(s)...`);
  
  try {
    const res = await fetch("/api/llm/job/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: filesToPolish }),
    });
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to start polish");
    }
    
    showToast(`LLM polish started for ${filesToPolish.length} file(s)`, "info");
    if (state.pipeline.afterOcr === "llm" && !state.transitionedToLlm) {
      beginLlmTransition(filesToPolish, true);
    }
    return filesToPolish;
  } catch (err) {
    appendLog(`[ui] Auto-polish error: ${err.message}`);
    showToast(`Auto-polish failed: ${err.message}`, "error");
    return null;
  }
}

// =============================================================================
// Session Persistence + Transition
// =============================================================================

function saveSessionState() {
  const session = {
    inputDir: state.inputDir || el("folderPath").value.trim(),
    outputDir: el("outputPath").value.trim() || state.outputDir || state.pipeline.outputDir,
    maxPages: state.maxPages,
    autoScroll: state.autoScroll,
    sortColumn: state.sortColumn,
    sortAsc: state.sortAsc,
  };
  StateManager.setJSON(StateManager.KEYS.SESSION, session);
}

function loadSessionState() {
  const session = StateManager.getJSON(StateManager.KEYS.SESSION, null);
  if (!session) return;

  if (session.inputDir) {
    state.inputDir = session.inputDir;
    const folderEl = el("folderPath");
    if (folderEl && !folderEl.value.trim()) folderEl.value = session.inputDir;
  }

  if (session.outputDir) {
    state.outputDir = session.outputDir;
    if (!state.pipeline.outputDir) state.pipeline.outputDir = session.outputDir;
    const outEl = el("outputPath");
    if (outEl && !outEl.value.trim()) outEl.value = session.outputDir;
  }

  if (typeof session.maxPages === "number") {
    state.maxPages = session.maxPages;
    const maxEl = el("maxPages");
    if (maxEl) maxEl.value = String(session.maxPages);
  }

  if (typeof session.autoScroll === "boolean") {
    state.autoScroll = session.autoScroll;
    const autoEl = el("autoScroll");
    if (autoEl) autoEl.checked = session.autoScroll;
  }

  if (session.sortColumn) state.sortColumn = session.sortColumn;
  if (typeof session.sortAsc === "boolean") state.sortAsc = session.sortAsc;
}

function showTransitionOverlay(message) {
  const overlay = el("transitionOverlay");
  const text = el("transitionMessage");
  if (!overlay || !text) return;
  text.textContent = message || "Switching to LLM…";
  overlay.classList.add("active");
}

function beginLlmTransition(files, autoStart) {
  const outputDir = el("outputPath").value.trim() || state.outputDir || state.pipeline.outputDir;
  state.transitionedToLlm = true;
  StateManager.setTransition({
    from: "ocr",
    outputDir,
    files: files || [],
    autoStart: Boolean(autoStart),
  });
  showTransitionOverlay("Switching to LLM…");
  setTimeout(() => {
    window.location.href = "/llm-polish";
  }, 400);
}

async function restoreJobStatus() {
  try {
    const res = await fetch("/api/job/status");
    const data = await res.json();
    if (!data.running) return;

    state.jobRunning = true;
    el("btnRun").disabled = true;
    el("btnStop").disabled = false;

    if (data.input_dir && !state.inputDir) state.inputDir = data.input_dir;
    if (data.output_dir && !state.outputDir) state.outputDir = data.output_dir;

    if (data.progress) {
      const p = data.progress;
      state.progress.fileIndex = p.file_index || 0;
      state.progress.totalFiles = p.total_files || 0;
      state.progress.currentFile = p.current_file || "";
      const item = state.items.find((it) => it.name === state.progress.currentFile);
      state.progress.filePages = item && item.pages ? item.pages : 0;
      state.progress.pagesDone = p.pages_done || 0;
      state.progress.totalPages = p.total_pages || 0;
      state.progress.speed = p.speed || 0;
      state.progress.vram = p.vram || 0;
      state.progress.eta = p.eta || "";
      state.progress.jobSpeed = p.job_speed || p.speed || 0;
      state.progress.jobEta = p.job_eta || p.eta || "";
      state.progress.batch_size = p.batch_size || 0;
      state.progress.stageName = p.stage_name || "";
      state.progress.stageCurrent = p.stage_current || 0;
      state.progress.stageTotal = p.stage_total || 0;
      state.progress.stageSpeed = p.stage_speed || 0;
      state.progress.stageEta = p.stage_eta || "";
    }

    updateJobStatus();
    showMonitorPanel();
    showOcrPreviewPanel();
    updateMonitorFromProgress();

    if (state.progress.stageName) {
      updateOcrPreviewLive(
        state.progress.stageName,
        state.progress.stageCurrent,
        state.progress.stageTotal
      );
    }

    appendLog("[ui] Restored running OCR job");
  } catch {}
}

async function restoreLlmJobStatus() {
  try {
    const res = await fetch("/api/llm/job/status");
    const data = await res.json();
    if (!data.running) return;

    if (state.pipeline.afterOcr === "llm") {
      beginLlmTransition(data.files || [], false);
      return;
    }

    showToast("LLM polish is running. Open LLM view to monitor.", "info");
  } catch {}
}

// =============================================================================
// Performance Settings (simplified - fixed config)
// =============================================================================

function loadSettings() {
  const saved = localStorage.getItem("ocr-settings");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      if (typeof parsed.autoFallback === "boolean") {
        state.settings.autoFallback = parsed.autoFallback;
      }
    } catch {}
  }
  applySettingsToUI();
}

function saveSettings() {
  localStorage.setItem("ocr-settings", JSON.stringify(state.settings));
}

function applySettingsToUI() {
  const fallbackEl = el("autoFallback");
  if (fallbackEl) {
    fallbackEl.checked = state.settings.autoFallback;
  }
}

function openSettingsModal() {
  el("settingsModal").hidden = false;
  applySettingsToUI();
  pollVram(); // Initial fetch
}

function closeSettingsModal() {
  el("settingsModal").hidden = true;
}

async function pollVram() {
  try {
    const res = await fetch("/api/vram");
    const data = await res.json();
    if (data.available && data.vram_total > 0) {
      const pct = (data.vram_used / data.vram_total) * 100;
      el("vramBar").style.width = `${pct}%`;
      el("vramText").textContent = `${data.vram_used.toFixed(1)}/${data.vram_total.toFixed(0)} GB`;
      el("gpuName").textContent = data.name || "GPU";
    } else {
      el("gpuName").textContent = "GPU: N/A";
      el("vramText").textContent = "--/-- GB";
    }
  } catch {
    el("gpuName").textContent = "GPU: N/A";
    el("vramText").textContent = "--/-- GB";
  }
}

// =============================================================================
// GPU Monitoring Panel (visible during job)
// =============================================================================

function showMonitorPanel() {
  el("gpuMonitor").classList.add("visible");
}

function hideMonitorPanel() {
  el("gpuMonitor").classList.remove("visible");
}

function updateMonitorPanel(data) {
  // Show panel if job is running
  if (state.jobRunning) {
    showMonitorPanel();
  }
  
  // GPU utilization
  const gpuUtil = data.gpu_util ?? -1;
  if (gpuUtil >= 0) {
    el("monitorGpuBar").style.width = `${gpuUtil}%`;
    el("monitorGpuUtil").textContent = `${gpuUtil}%`;
    // Add warning class if high
    if (gpuUtil > 90) {
      el("monitorGpuBar").classList.add("high");
    } else {
      el("monitorGpuBar").classList.remove("high");
    }
  }
  
  // VRAM
  const vramUsed = data.vram_used ?? 0;
  const vramTotal = data.vram_total ?? 16;
  if (vramTotal > 0) {
    const vramPct = (vramUsed / vramTotal) * 100;
    el("monitorVramBar").style.width = `${vramPct}%`;
    el("monitorVramText").textContent = `${vramUsed.toFixed(1)}/${vramTotal.toFixed(0)} GB`;
    // Add critical class if > 95%
    if (vramPct > 95) {
      el("monitorVramBar").classList.add("critical");
    } else {
      el("monitorVramBar").classList.remove("critical");
    }
  }
  
  // Temperature
  const temp = data.gpu_temp ?? -1;
  if (temp >= 0) {
    el("monitorTemp").textContent = `${temp}°C`;
    // Add hot class if > 80°C
    if (temp > 80) {
      el("monitorTemp").classList.add("hot");
    } else {
      el("monitorTemp").classList.remove("hot");
    }
  }
}

function updateMonitorFromProgress() {
  // Update speed, batch, ETA from progress state
  const p = state.progress;

  const speed = p.jobSpeed > 0 ? p.jobSpeed : p.speed;
  if (speed > 0) {
    el("monitorSpeed").textContent = `${speed.toFixed(1)} p/min`;
  }
  
  if (p.batch_size > 0) {
    el("monitorBatch").textContent = String(p.batch_size);
  }
  
  const eta = p.jobEta || p.eta;
  if (eta) {
    el("monitorEta").textContent = eta;
  }
}

// =============================================================================
// OCR Preview Panel
// =============================================================================

function showOcrPreviewPanel() {
  el("ocrPreviewPanel").hidden = false;
}

function hideOcrPreviewPanel() {
  el("ocrPreviewPanel").hidden = true;
}

function updateOcrPreviewLive(stage, current, total) {
  // Show live progress during OCR
  showOcrPreviewPanel();
  const percent = total > 0 ? Math.round((current / total) * 100) : 0;
  const pagesInfo = state.progress.filePages > 0
    ? `File pages: ${state.progress.filePages}`
    : "";
  const stageInfo = total > 0
    ? `Stage progress: ${current}/${total} (${percent}%)`
    : `Stage progress: ${current}`;
  el("ocrPreviewInfo").textContent = `${stage} • ${stageInfo}`;
  el("ocrPreviewText").textContent = `Processing... ${stage}\n\n${stageInfo}${pagesInfo ? "\n" + pagesInfo : ""}\n\nOCR text will appear here when file completes.`;
}

function updateOcrPreview(data) {
  // Show panel
  showOcrPreviewPanel();
  
  // Update info line
  const info = `${data.filename} • ${data.pages} pages • ${(data.total_chars / 1000).toFixed(1)}K chars`;
  el("ocrPreviewInfo").textContent = info;
  
  // Update preview text (unescape \\n to actual newlines)
  let previewText = data.preview || "";
  previewText = previewText.replace(/\\n/g, "\n");
  el("ocrPreviewText").textContent = previewText;
}

// =============================================================================
// Toast Notifications
// =============================================================================

function showToast(message, type = "info") {
  const container = el("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(20px)";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// =============================================================================
// Status Display
// =============================================================================

function updateJobStatus() {
  const pill = el("jobStatus");
  const bar = el("progressBar");
  
  if (!state.jobRunning) {
    pill.textContent = "idle";
    pill.classList.remove("running");
    bar.style.width = "0%";
    return;
  }
  
  pill.classList.add("running");
  
  const p = state.progress;
  if (p.totalPages > 0) {
    const percent = Math.round((p.pagesDone / p.totalPages) * 100);
    bar.style.width = `${percent}%`;
    
    let text = `${percent}%`;
    if (p.currentFile) {
      const shortName = p.currentFile.length > 25 
        ? p.currentFile.slice(0, 22) + "..." 
        : p.currentFile;
      text += ` • ${shortName}`;
    }
    pill.textContent = text;
  } else if (p.totalFiles > 0) {
    pill.textContent = `${p.fileIndex}/${p.totalFiles}`;
    bar.style.width = `${(p.fileIndex / p.totalFiles) * 100}%`;
  } else {
    pill.textContent = "running...";
    bar.style.width = "10%";
  }
}

function updateStats() {
  const stats = el("stats");
  if (!state.items.length) {
    stats.textContent = "";
    return;
  }
  
  const total = state.items.length;
  const done = state.items.filter(i => i.done).length;
  const pending = state.items.filter(i => !i.done && !i.too_long).length;
  const selected = state.selection.size;
  
  stats.textContent = `${selected} selected • ${pending} pending • ${done}/${total} done`;
}

// =============================================================================
// Log Panel
// =============================================================================

function appendLog(line) {
  const log = el("log");
  
  // Apply syntax highlighting
  let html = escapeHtml(line);
  
  if (line.includes("DONE:") || line.includes("COMPLETE")) {
    html = `<span class="log-success">${html}</span>`;
  } else if (line.includes("FAILED:") || line.includes("Error") || line.includes("error")) {
    html = `<span class="log-error">${html}</span>`;
  } else if (line.includes("[server]") || line.includes("[ui]")) {
    html = `<span class="log-muted">${html}</span>`;
  } else if (line.includes("Progress:") || line.includes("Speed:")) {
    html = `<span class="log-info">${html}</span>`;
  } else if (line.includes("batch=") || line.includes("VRAM:")) {
    html = `<span class="log-warning">${html}</span>`;
  }
  
  log.innerHTML += html + "\n";
  
  if (state.autoScroll) {
    log.scrollTop = log.scrollHeight;
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function clearLog() {
  el("log").innerHTML = "";
}

// =============================================================================
// Sorting
// =============================================================================

function statusOrder(item) {
  if (item.done) return 2;
  if (item.too_long) return 3;
  return 1; // pending first
}

function sortItems() {
  state.items.sort((a, b) => {
    let cmp = 0;
    switch (state.sortColumn) {
      case "name":
        cmp = a.name.localeCompare(b.name);
        break;
      case "pages":
        cmp = a.pages - b.pages;
        break;
      case "status":
        cmp = statusOrder(a) - statusOrder(b);
        break;
    }
    return state.sortAsc ? cmp : -cmp;
  });
}

function handleSort(column) {
  if (state.sortColumn === column) {
    state.sortAsc = !state.sortAsc;
  } else {
    state.sortColumn = column;
    state.sortAsc = true;
  }
  sortItems();
  renderTable();
  updateSortHeaders();
  saveSessionState();
}

function updateSortHeaders() {
  $$("th.sortable").forEach(th => {
    th.classList.remove("asc", "desc");
    if (th.dataset.col === state.sortColumn) {
      th.classList.add(state.sortAsc ? "asc" : "desc");
    }
  });
}

// =============================================================================
// Table Rendering
// =============================================================================

function fmtStatus(it) {
  if (it.polished) return { text: "Polished", cls: "polished" };
  if (it.done) return { text: "Done", cls: "done" };
  if (it.too_long) return { text: "Skip", cls: "skip" };
  return { text: "Pending", cls: "pending" };
}

function renderTable() {
  const tbody = el("pdfTbody");
  tbody.innerHTML = "";

  for (const it of state.items) {
    const tr = document.createElement("tr");
    const isDisabled = it.done || it.too_long;
    if (isDisabled) tr.classList.add("disabled");

    // Checkbox
    const tdC = document.createElement("td");
    tdC.className = "colCheck";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selection.has(it.path);
    cb.disabled = isDisabled;
    cb.addEventListener("change", () => {
      if (cb.checked) state.selection.add(it.path);
      else state.selection.delete(it.path);
      updateStats();
      updateCheckAll();
    });
    tdC.appendChild(cb);

    // File name
    const tdN = document.createElement("td");
    const nameSpan = document.createElement("span");
    nameSpan.className = "file-name";
    nameSpan.textContent = it.name;
    tdN.appendChild(nameSpan);

    // Pages
    const tdP = document.createElement("td");
    tdP.className = "colPages";
    tdP.textContent = it.pages || "-";

    // Status
    const tdS = document.createElement("td");
    tdS.className = "colStatus";
    const st = fmtStatus(it);
    const span = document.createElement("span");
    span.className = `status ${st.cls}`;
    span.textContent = st.text;
    tdS.appendChild(span);

    tr.appendChild(tdC);
    tr.appendChild(tdN);
    tr.appendChild(tdP);
    tr.appendChild(tdS);
    tbody.appendChild(tr);
  }
  
  // Update footer
  const footer = el("tableFooter");
  if (state.items.length) {
    const totalPages = state.items.reduce((s, i) => s + (i.pages || 0), 0);
    footer.textContent = `${state.items.length} files • ${totalPages.toLocaleString()} total pages`;
  } else {
    footer.textContent = "No PDF files found";
  }
  
  updateStats();
  updateCheckAll();
}

function updateCheckAll() {
  const checkAll = el("checkAll");
  const selectable = state.items.filter(i => !i.done && !i.too_long);
  if (selectable.length === 0) {
    checkAll.checked = false;
    checkAll.indeterminate = false;
  } else if (state.selection.size === 0) {
    checkAll.checked = false;
    checkAll.indeterminate = false;
  } else if (state.selection.size === selectable.length) {
    checkAll.checked = true;
    checkAll.indeterminate = false;
  } else {
    checkAll.checked = false;
    checkAll.indeterminate = true;
  }
}

// =============================================================================
// Selection
// =============================================================================

function selectAll() {
  state.selection.clear();
  for (const it of state.items) {
    if (!it.done && !it.too_long) state.selection.add(it.path);
  }
  renderTable();
}

function selectNone() {
  state.selection.clear();
  renderTable();
}

function selectPending() {
  state.selection.clear();
  for (const it of state.items) {
    if (!it.done && !it.too_long) state.selection.add(it.path);
  }
  renderTable();
}

function toggleSelectAll() {
  const selectable = state.items.filter(i => !i.done && !i.too_long);
  if (state.selection.size === selectable.length) {
    selectNone();
  } else {
    selectAll();
  }
}

// =============================================================================
// API Calls
// =============================================================================

async function scan() {
  let dir = el("folderPath").value.trim();
  const maxPages = parseInt(el("maxPages").value) || 500;
  state.maxPages = maxPages;
  if (!dir && state.inputDir) {
    dir = state.inputDir;
    el("folderPath").value = dir;
  }
  
  if (!dir) {
    showToast("Please select a folder first", "error");
    return;
  }

  updateJobStatus();
  appendLog("[ui] Scanning folder...");

  try {
    const res = await fetch(`/api/scan?dir=${encodeURIComponent(dir)}&max_pages=${maxPages}`);
    const data = await res.json();
    
    if (data.error) {
      appendLog(`[ui] Scan error: ${data.error}`);
      showToast(data.error, "error");
      return;
    }
    
    state.inputDir = data.input_dir;
    state.outputDir = data.output_dir;
    state.items = data.items || [];
    saveSessionState();
    
    // Update pipeline output if not set
    if (!el("outputPath").value.trim()) {
      el("outputPath").value = data.output_dir;
      savePipelineSettings();
    }

    // Default selection: pending items
    state.selection.clear();
    for (const it of state.items) {
      if (!it.done && !it.too_long) state.selection.add(it.path);
    }

    sortItems();
    renderTable();
    updateSortHeaders();
    
    const pending = state.items.filter(i => !i.done && !i.too_long).length;
    appendLog(`[ui] Found ${state.items.length} PDFs, ${pending} pending`);
    
  } catch (err) {
    appendLog(`[ui] Scan failed: ${err.message}`);
    showToast("Scan failed", "error");
  }
}

async function startJob() {
  if (state.selection.size === 0) {
    showToast("No files selected", "error");
    return;
  }

  saveSessionState();
  
  el("btnRun").disabled = true;
  el("btnStop").disabled = false;
  state.jobRunning = true;
  state.transitionedToLlm = false;
  state.progress = {
    fileIndex: 0,
    totalFiles: state.selection.size,
    currentFile: "",
    filePages: 0,
    pagesDone: 0,
    totalPages: 0,
    speed: 0,
    vram: 0,
    eta: "",
    jobSpeed: 0,
    jobEta: "",
    stageName: "",
    stageCurrent: 0,
    stageTotal: 0,
    stageSpeed: 0,
    stageEta: "",
    batch_size: 0,
  };
  updateJobStatus();
  showMonitorPanel(); // Show monitoring panel when job starts
  showOcrPreviewPanel(); // Show OCR preview panel
  el("ocrPreviewText").textContent = "Waiting for OCR output...";
  el("ocrPreviewInfo").textContent = "Processing...";

  const payload = {
    input_dir: state.inputDir,
    output_dir: el("outputPath").value.trim() || state.outputDir,
    max_pages: state.maxPages,
    files: Array.from(state.selection),
    auto_fallback: state.settings.autoFallback,
  };

  try {
    const res = await fetch("/api/job/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Start failed");
    }
    
    showToast(`Started OCR for ${state.selection.size} files`, "success");
    
  } catch (err) {
    appendLog(`[ui] Start failed: ${err.message}`);
    showToast(err.message, "error");
    el("btnRun").disabled = false;
    el("btnStop").disabled = true;
    state.jobRunning = false;
    updateJobStatus();
  }
}

async function stopJob() {
  el("btnStop").disabled = true;
  
  try {
    const res = await fetch("/api/job/stop", { method: "POST" });
    if (!res.ok) {
      throw new Error("Stop failed");
    }
    showToast("Job stopped", "info");
  } catch (err) {
    appendLog(`[ui] Stop failed: ${err.message}`);
    showToast("Stop failed", "error");
  }
  
  el("btnRun").disabled = false;
  state.jobRunning = false;
  updateJobStatus();
}

// =============================================================================
// SSE Event Handling
// =============================================================================

function connectSSE() {
  if (state.sse) {
    state.sse.close();
  }
  
  const es = new EventSource("/api/job/stream");
  state.sse = es;
  
  es.addEventListener("ready", () => {
    appendLog("[ui] Connected to server");
  });
  
  es.addEventListener("log", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      appendLog(data.line);
    } catch {
      appendLog(ev.data);
    }
  });
  
  es.addEventListener("file_start", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.fileIndex = data.file_index;
      state.progress.totalFiles = data.total_files;
      state.progress.currentFile = data.current_file;
      const item = state.items.find((it) => it.name === state.progress.currentFile);
      state.progress.filePages = item && item.pages ? item.pages : 0;
      updateJobStatus();
    } catch {}
  });
  
  es.addEventListener("progress", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.pagesDone = data.pages_done;
      state.progress.totalPages = data.total_pages;
      updateJobStatus();
    } catch {}
  });
  
  es.addEventListener("file_done", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.speed = data.speed || 0;
      state.progress.vram = data.vram || 0;
      state.progress.eta = data.eta || "";
      state.progress.jobSpeed = data.job_speed || data.speed || 0;
      state.progress.jobEta = data.job_eta || data.eta || "";
      state.progress.batch_size = data.batch_size || 0;
      if (data.file_pages) state.progress.filePages = data.file_pages;
      updateJobStatus();
      updateMonitorFromProgress(); // Update monitor panel with speed/batch/eta
      
      // Mark file as done in local state
      const file = state.items.find(i => i.name === data.current_file || i.path.endsWith(data.current_file));
      if (file) {
        file.done = true;
        state.selection.delete(file.path);
        renderTable();
      }
      
      // Auto-polish trigger
      if (state.pipeline.llmMode === "after_each") {
        triggerLlmPolish([data.current_file]);
      }
    } catch {}
  });
  
  es.addEventListener("metrics_update", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.speed > 0) state.progress.speed = data.speed;
      if (data.vram > 0) state.progress.vram = data.vram;
      if (data.eta) state.progress.eta = data.eta;
      if (data.job_speed > 0) state.progress.jobSpeed = data.job_speed;
      if (data.job_eta) state.progress.jobEta = data.job_eta;
      if (!state.progress.jobSpeed && data.speed > 0) state.progress.jobSpeed = data.speed;
      if (!state.progress.jobEta && data.eta) state.progress.jobEta = data.eta;
      if (data.batch_size > 0) state.progress.batch_size = data.batch_size;
      updateMonitorFromProgress();
    } catch {}
  });
  
  es.addEventListener("tqdm_progress", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.stage) state.progress.stageName = data.stage;
      if (data.stage_name) state.progress.stageName = data.stage_name;
      if (data.stage_current >= 0) state.progress.stageCurrent = data.stage_current || 0;
      if (data.stage_total >= 0) state.progress.stageTotal = data.stage_total || 0;
      if (data.stage_speed >= 0) state.progress.stageSpeed = data.stage_speed || 0;
      if (data.stage_eta) state.progress.stageEta = data.stage_eta;
      updateMonitorFromProgress();
      updateJobStatus();
      
      // Update OCR preview with live progress
      const liveStage = state.progress.stageName || data.stage;
      if (liveStage) {
        updateOcrPreviewLive(liveStage, state.progress.stageCurrent, state.progress.stageTotal);
      }
    } catch {}
  });
  
  es.addEventListener("file_failed", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      showToast(`Failed: ${data.file}`, "error");
    } catch {}
  });
  
  es.addEventListener("job_complete", async (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.jobRunning = false;
      el("btnRun").disabled = false;
      el("btnStop").disabled = true;
      updateJobStatus();
      hideMonitorPanel();
      
      if (data.success) {
        showToast("OCR completed successfully!", "success");
      } else {
        showToast("OCR finished with errors", "error");
      }
      
      // Auto-polish trigger (batch)
      if (state.pipeline.llmMode === "after_all") {
        appendLog("[ui] Job complete. Triggering batch polish...");
        const started = await triggerLlmPolish([]);
        if (started && state.pipeline.afterOcr === "llm" && !state.transitionedToLlm) {
          beginLlmTransition(started, true);
          return;
        }
      }
      
      // Refresh to get accurate status
      setTimeout(() => scan(), 500);
    } catch {}
  });
  
  es.addEventListener("job_stopped", () => {
    state.jobRunning = false;
    el("btnRun").disabled = false;
    el("btnStop").disabled = true;
    updateJobStatus();
    hideMonitorPanel();
  });
  
  // GPU stats from subprocess
  es.addEventListener("gpu_stats", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      updateMonitorPanel(data);
    } catch {}
  });
  
  // OCR text preview
  es.addEventListener("ocr_preview", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      updateOcrPreview(data);
    } catch {}
  });
  
  es.onerror = () => {
    // Browser will auto-reconnect
  };
}

// =============================================================================
// Folder Browser Modal
// =============================================================================

async function browse(path) {
  const errorEl = el("browseError");
  errorEl.classList.remove("visible");
  
  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(path || "")}`);
    const data = await res.json();
    
    if (data.error) {
      errorEl.textContent = data.error;
      errorEl.classList.add("visible");
      return;
    }
    
    el("browsePath").value = data.path;
    renderBreadcrumbs(data.path);
    
    const list = el("browseList");
    list.innerHTML = "";

    const folders = (data.items || []).filter(i => i.is_dir);
    
    if (folders.length === 0) {
      const empty = document.createElement("div");
      empty.className = "browseItem";
      empty.innerHTML = '<span class="folder-name" style="color: var(--muted)">No subfolders</span>';
      list.appendChild(empty);
      return;
    }

    for (const it of folders) {
      const row = document.createElement("div");
      row.className = "browseItem";

      const icon = document.createElement("span");
      icon.className = "folder-icon";
      icon.textContent = "📁";

      const name = document.createElement("span");
      name.className = "folder-name";
      name.textContent = it.name;

      const actions = document.createElement("div");
      actions.className = "actions";

      const openBtn = document.createElement("button");
      openBtn.className = "link-btn";
      openBtn.type = "button";
      openBtn.textContent = "Open";
      openBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        browse(it.path);
      });

      const chooseBtn = document.createElement("button");
      chooseBtn.className = "link-btn";
      chooseBtn.type = "button";
      chooseBtn.textContent = "Choose";
      chooseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (state.browsingTarget === "input") {
          el("folderPath").value = it.path;
          scan();
        } else {
          el("outputPath").value = it.path;
          savePipelineSettings();
        }
        closeModal();
      });

      actions.appendChild(openBtn);
      actions.appendChild(chooseBtn);

      row.appendChild(icon);
      row.appendChild(name);
      row.appendChild(actions);

      row.addEventListener("dblclick", () => browse(it.path));
      list.appendChild(row);
    }
  } catch (err) {
    errorEl.textContent = `Error: ${err.message}`;
    errorEl.classList.add("visible");
  }
}

function renderBreadcrumbs(path) {
  const container = el("breadcrumbs");
  container.innerHTML = "";
  
  // Handle Windows paths
  const parts = path.split(/[\\/]/).filter(Boolean);
  let accumulated = "";
  
  // Windows drive letter handling
  const isWindows = /^[A-Za-z]:/.test(path);
  
  parts.forEach((part, i) => {
    if (i === 0 && isWindows) {
      // For drive letter, include the backslash (C:\)
      accumulated = part + "\\";
    } else {
      accumulated += (accumulated.endsWith("\\") || accumulated.endsWith(":") || accumulated === "" ? "" : "\\") + part;
    }
    
    const crumb = document.createElement("span");
    crumb.className = "crumb";
    crumb.textContent = part;
    const pathForThisCrumb = accumulated;  // Capture current value, not reference
    crumb.addEventListener("click", () => browse(pathForThisCrumb));
    container.appendChild(crumb);
    
    if (i < parts.length - 1) {
      const sep = document.createElement("span");
      sep.className = "crumb-sep";
      sep.textContent = "›";
      container.appendChild(sep);
    }
  });
}

function openModal(target = "input") {
  state.browsingTarget = target;
  el("modal").hidden = false;
  
  let currentPath = "";
  if (target === "input") {
    currentPath = el("folderPath").value.trim();
  } else {
    currentPath = el("outputPath").value.trim();
  }
  
  browse(currentPath || "");
}

function closeModal() {
  el("modal").hidden = true;
}

function useModalPath() {
  const p = el("browsePath").value.trim();
  if (!p) return;
  
  if (state.browsingTarget === "input") {
    el("folderPath").value = p;
    scan();
  } else {
    el("outputPath").value = p;
    savePipelineSettings();
    saveSessionState();
  }
  closeModal();
}

async function goUp() {
  const p = el("browsePath").value.trim();
  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(p)}`);
    const data = await res.json();
    if (data.parent) {
      browse(data.parent);
    }
  } catch {}
}

// =============================================================================
// Post-processing Functions
// =============================================================================

async function checkLlmStatus() {
  const statusEl = el("llmStatus");
  const ind = el("llmStatusIndicator");
  const txt = el("llmStatusText");
  
  try {
    const res = await fetch("/api/postprocess/llm-status");
    const data = await res.json();
    state.pipeline.llmAvailable = data.available;
    
    if (data.available) {
      if (statusEl) {
        statusEl.textContent = "LM Studio: online";
        statusEl.className = "llm-status online";
      }
      if (ind) {
        ind.className = "llm-status-indicator online";
        txt.textContent = "Online";
      }
    } else {
      if (statusEl) {
        statusEl.textContent = "LM Studio: offline";
        statusEl.className = "llm-status offline";
      }
      if (ind) {
        ind.className = "llm-status-indicator offline";
        txt.textContent = "Offline";
      }
    }
  } catch {
    state.pipeline.llmAvailable = false;
    if (statusEl) {
      statusEl.textContent = "LM Studio: offline";
      statusEl.className = "llm-status offline";
    }
    if (ind) {
      ind.className = "llm-status-indicator offline";
      txt.textContent = "Offline";
    }
  }
}

function openPostprocessModal(title) {
  el("postprocessTitle").textContent = title;
  el("postprocessStatus").className = "postprocess-status";
  el("postprocessStatus").innerHTML = '<div class="spinner"></div><span>Processing...</span>';
  el("postprocessOutput").textContent = "";
  el("postprocessModal").hidden = false;
}

function closePostprocessModal() {
  el("postprocessModal").hidden = true;
}

function updatePostprocessStatus(success, message) {
  const statusEl = el("postprocessStatus");
  if (success) {
    statusEl.className = "postprocess-status success";
    statusEl.innerHTML = `<span>${message}</span>`;
  } else {
    statusEl.className = "postprocess-status error";
    statusEl.innerHTML = `<span>${message}</span>`;
  }
}

async function runFixMojibake() {
  // Get list of done files
  const doneFiles = state.items
    .filter(i => i.done)
    .map(i => {
      // Convert PDF path to MD path
      const pdfPath = i.path;
      const mdName = pdfPath.replace(/\.pdf$/i, ".md").split(/[\\/]/).pop();
      return (state.pipeline.outputDir || state.outputDir) + "\\" + mdName;
    });

  if (doneFiles.length === 0) {
    showToast("No converted files to process", "error");
    return;
  }

  openPostprocessModal("Fix Mojibake");

  try {
    const res = await fetch("/api/postprocess/fix-mojibake", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: doneFiles }),
    });

    const data = await res.json();

    if (data.ok) {
      let output = "";
      let fixedCount = 0;
      for (const r of data.results || []) {
        if (r.success) {
          output += r.output + "\n";
          if (r.output && r.output.includes("FIXED:")) {
            fixedCount++;
          }
        } else {
          output += `ERROR: ${r.file}: ${r.error}\n`;
        }
      }
      el("postprocessOutput").textContent = output || "No output";
      updatePostprocessStatus(true, `Processed ${doneFiles.length} files, fixed ${fixedCount}`);
      showToast("Mojibake fix complete", "success");
    } else {
      el("postprocessOutput").textContent = data.error || "Unknown error";
      updatePostprocessStatus(false, "Fix failed");
      showToast("Mojibake fix failed", "error");
    }
  } catch (err) {
    el("postprocessOutput").textContent = err.message;
    updatePostprocessStatus(false, "Request failed");
    showToast("Mojibake fix failed", "error");
  }
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
  // Theme
  initTheme();
  el("btnTheme").addEventListener("click", toggleTheme);
  
  // Settings modal
  el("btnSettings").addEventListener("click", openSettingsModal);
  el("btnCloseSettings").addEventListener("click", closeSettingsModal);
  
  // Settings modal backdrop
  const settingsBackdrop = document.querySelector("#settingsModal .modal-backdrop");
  if (settingsBackdrop) {
    settingsBackdrop.addEventListener("click", closeSettingsModal);
  }
  
  // Auto-fallback checkbox
  const autoFallbackEl = el("autoFallback");
  if (autoFallbackEl) {
    autoFallbackEl.addEventListener("change", (e) => {
      state.settings.autoFallback = e.target.checked;
      saveSettings();
    });
  }
  
  // Load saved settings
  loadSettings();
  loadPipelineSettings();
  loadSessionState();
  updateSortHeaders();
  
  // Pipeline settings listeners
  el("outputPath").addEventListener("change", () => {
    savePipelineSettings();
    saveSessionState();
  });
  el("btnBrowseOutput").addEventListener("click", () => openModal("output"));
  
  const radios = document.getElementsByName("llmMode");
  for (const r of radios) {
    r.addEventListener("change", () => {
      savePipelineSettings();
      saveSessionState();
    });
  }

  const afterRadios = document.getElementsByName("afterOcr");
  for (const r of afterRadios) {
    r.addEventListener("change", () => {
      savePipelineSettings();
      saveSessionState();
    });
  }
  
  // Poll VRAM every 2 seconds when settings modal is open
  setInterval(() => {
    if (!el("settingsModal").hidden) pollVram();
  }, 2000);
  
  // Check LLM status periodically
  checkLlmStatus();
  setInterval(checkLlmStatus, 30000);
  
  // Post-processing buttons
  el("btnFixMojibake").addEventListener("click", runFixMojibake);
  el("btnClosePostprocess").addEventListener("click", closePostprocessModal);
  
  // Post-process modal backdrop
  const ppBackdrop = document.querySelector("#postprocessModal .modal-backdrop");
  if (ppBackdrop) {
    ppBackdrop.addEventListener("click", closePostprocessModal);
  }
  
  // Folder controls
  el("btnBrowse").addEventListener("click", () => openModal("input"));
  el("btnScan").addEventListener("click", scan);
  
  // Selection buttons
  el("btnSelectAll").addEventListener("click", selectAll);
  el("btnSelectNone").addEventListener("click", selectNone);
  el("btnSelectNew").addEventListener("click", selectPending);
  el("checkAll").addEventListener("change", toggleSelectAll);
  
  // Job controls
  el("btnRun").addEventListener("click", startJob);
  el("btnStop").addEventListener("click", stopJob);
  
  // Log controls
  el("btnClearLog").addEventListener("click", clearLog);
  el("autoScroll").addEventListener("change", (e) => {
    state.autoScroll = e.target.checked;
    saveSessionState();
  });
  
  // Modal controls
  el("btnCloseModal").addEventListener("click", closeModal);
  el("btnUseFolder").addEventListener("click", useModalPath);
  el("btnGo").addEventListener("click", () => browse(el("browsePath").value.trim()));
  el("btnUp").addEventListener("click", goUp);
  
  // Modal backdrop click
  $(".modal-backdrop").addEventListener("click", closeModal);
  
  // Keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!el("postprocessModal").hidden) {
        closePostprocessModal();
      } else if (!el("settingsModal").hidden) {
        closeSettingsModal();
      } else if (!el("modal").hidden) {
        closeModal();
      }
    }
  });
  
  el("browsePath").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      browse(el("browsePath").value.trim());
    }
  });
  
  el("folderPath").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      scan();
    }
  });

  el("folderPath").addEventListener("change", () => {
    state.inputDir = el("folderPath").value.trim();
    saveSessionState();
  });
  
  // Sortable headers
  $$("th.sortable").forEach(th => {
    th.addEventListener("click", () => handleSort(th.dataset.col));
  });
  
  // Max pages input
  el("maxPages").addEventListener("change", () => {
    state.maxPages = parseInt(el("maxPages").value) || 500;
    saveSessionState();
  });
  
  // Connect SSE
  connectSSE();
  
  // Load initial folder (prefer saved session)
  const initialPath = el("folderPath").value.trim();
  if (initialPath) {
    await scan();
  } else {
    try {
      const d = await fetch("/api/browse").then(r => r.json());
      el("folderPath").value = d.path;
      await scan();
    } catch {
      el("folderPath").value = "";
    }
  }

  await restoreJobStatus();
  await restoreLlmJobStatus();
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((err) => {
    console.error(err);
  });
});
