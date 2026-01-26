/* =============================================================================
   LLM Polish Page - Application Logic
   ============================================================================= */

// =============================================================================
// State
// =============================================================================

const state = {
  items: [],           // MD files list
  selection: new Set(),
  outputDir: "",
  sse: null,
  
  // Job state
  jobRunning: false,
  progress: {
    fileIndex: 0,
    totalFiles: 0,
    currentFile: "",
    chunkIndex: 0,
    totalChunks: 0,
    speed: 0,
    eta: "",
    filesModified: 0,
  },
  
  // Sorting
  sortColumn: "name",
  sortAsc: true,
  
  // UI
  autoScroll: true,
  theme: "system",
  
  // LM Studio status
  llmAvailable: false,
  llmModels: [],
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
  
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.theme === "system") applyTheme();
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
// Session Persistence + Transition
// =============================================================================

function saveSessionState(partial) {
  const session = StateManager.getJSON(StateManager.KEYS.SESSION, {});
  const next = { ...session, ...partial };
  StateManager.setJSON(StateManager.KEYS.SESSION, next);
}

function loadSessionState() {
  const session = StateManager.getJSON(StateManager.KEYS.SESSION, null);
  if (!session) return;

  if (session.outputDir) state.outputDir = session.outputDir;
  if (typeof session.autoScroll === "boolean") state.autoScroll = session.autoScroll;
  if (session.sortColumn) state.sortColumn = session.sortColumn;
  if (typeof session.sortAsc === "boolean") state.sortAsc = session.sortAsc;

  const autoEl = el("autoScroll");
  if (autoEl) autoEl.checked = state.autoScroll;
}

async function restoreLlmJobStatus() {
  try {
    const res = await fetch("/api/llm/job/status");
    const data = await res.json();
    if (!data.running) return;

    state.jobRunning = true;
    el("btnRun").disabled = true;
    el("btnStop").disabled = false;

    if (data.progress) {
      const p = data.progress;
      state.progress.fileIndex = p.file_index || 0;
      state.progress.totalFiles = p.total_files || 0;
      state.progress.currentFile = p.current_file || "";
      state.progress.chunkIndex = p.chunk_index || 0;
      state.progress.totalChunks = p.total_chunks || 0;
      state.progress.speed = p.speed_chars_per_sec || p.speed || 0;
      state.progress.eta = p.eta || "";
      state.progress.filesModified = p.files_modified || 0;
    }

    updateJobStatus();
    el("progressCard").hidden = false;
    updateProgressPanel();
    appendLog("[ui] Restored running LLM polish job");
  } catch {}
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
// LM Studio Status
// =============================================================================

async function checkLlmStatus() {
  const indicator = el("llmStatusIndicator");
  const text = el("llmStatusText");
  const details = el("llmStatusDetails");
  
  indicator.className = "status-indicator checking";
  text.textContent = "Checking LM Studio...";
  
  try {
    const res = await fetch("/api/postprocess/llm-status");
    const data = await res.json();
    
    if (data.available) {
      state.llmAvailable = true;
      state.llmModels = data.models || [];
      indicator.className = "status-indicator online";
      text.textContent = "LM Studio Online";
      details.innerHTML = state.llmModels.length > 0 
        ? `Model: <span class="model-name">${state.llmModels[0]}</span>`
        : "No model loaded";
      el("btnRun").disabled = state.selection.size === 0;
    } else {
      state.llmAvailable = false;
      indicator.className = "status-indicator offline";
      text.textContent = "LM Studio Offline";
      details.textContent = data.error || "Start LM Studio and load a model";
      el("btnRun").disabled = true;
    }
  } catch (err) {
    state.llmAvailable = false;
    indicator.className = "status-indicator offline";
    text.textContent = "Connection Error";
    details.textContent = err.message;
    el("btnRun").disabled = true;
  }
}

// =============================================================================
// Status Display
// =============================================================================

function calculateOverallPercent(p) {
  // Calculate overall progress including current file's chunk progress
  // Formula: (completedFiles + currentFileChunkProgress) / totalFiles
  if (p.totalFiles <= 0) return 0;
  
  // Current file's partial progress (0 to 1)
  const currentFileProgress = p.totalChunks > 0 ? p.chunkIndex / p.totalChunks : 0;
  
  // Overall progress
  const overallProgress = (p.fileIndex + currentFileProgress) / p.totalFiles;
  
  return Math.round(overallProgress * 100);
}

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
  
  if (p.totalFiles > 0) {
    const percent = calculateOverallPercent(p);
    bar.style.width = `${percent}%`;
    pill.textContent = `${percent}% • ${p.currentFile || "..."}`;
  } else {
    pill.textContent = "running...";
    bar.style.width = "10%";
  }
}

function updateChunkDiff(original, result) {
  // Unescape newlines from JSON
  const origText = original.replace(/\\n/g, '\n');
  const resultText = result.replace(/\\n/g, '\n');
  
  el("diffOriginal").textContent = origText || "(empty)";
  el("diffResult").textContent = resultText || "(empty)";
}

function updateProgressPanel() {
  const p = state.progress;
  
  el("currentFile").textContent = p.currentFile || "-";
  
  // FILES: Show current file number (1-indexed for user-friendly display)
  // fileIndex is 0 until a file completes, so show (fileIndex + 1) when a file is being processed
  const currentFileNum = p.currentFile ? p.fileIndex + 1 : p.fileIndex;
  el("filesProgress").textContent = `${currentFileNum}/${p.totalFiles}`;
  
  el("chunksProgress").textContent = `${p.chunkIndex}/${p.totalChunks}`;
  el("speed").textContent = p.speed > 0 ? `${Math.round(p.speed)} chars/s` : "- chars/s";
  el("eta").textContent = p.eta || "--:--";
  el("modified").textContent = String(p.filesModified);
  
  // Chunk bar
  const chunkPercent = p.totalChunks > 0 ? (p.chunkIndex / p.totalChunks) * 100 : 0;
  el("chunkBar").style.width = `${chunkPercent}%`;
  
  // Summary - use the same calculation as header
  const filePercent = calculateOverallPercent(p);
  el("progressSummary").textContent = `${filePercent}% complete`;
}

function updateStats() {
  const stats = el("stats");
  if (!state.items.length) {
    stats.textContent = "";
    return;
  }
  
  const total = state.items.length;
  const polished = state.items.filter(i => i.polished).length;
  const unpolished = state.items.filter(i => !i.polished).length;
  const selected = state.selection.size;
  
  stats.textContent = `${selected} selected • ${polished} polished • ${unpolished} pending`;
}

// =============================================================================
// Log Panel
// =============================================================================

function appendLog(line) {
  const log = el("log");
  
  let html = escapeHtml(line);
  
  if (line.includes("SAVED:") || line.includes("COMPLETE")) {
    html = `<span class="log-success">${html}</span>`;
  } else if (line.includes("ERROR") || line.includes("FAILED")) {
    html = `<span class="log-error">${html}</span>`;
  } else if (line.includes("[server]")) {
    html = `<span class="log-muted">${html}</span>`;
  } else if (line.includes("Chunk") || line.includes("Processing:")) {
    html = `<span class="log-info">${html}</span>`;
  } else if (line.includes("WARNING")) {
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
// Table Rendering
// =============================================================================

function statusOrder(item) {
  if (item.polished) return 3;
  if (item.done) return 2;
  return 1;
}

function sortItems() {
  state.items.sort((a, b) => {
    let cmp = 0;
    switch (state.sortColumn) {
      case "name":
        cmp = a.name.localeCompare(b.name);
        break;
      case "size":
        cmp = (a.size_kb || 0) - (b.size_kb || 0);
        break;
      case "status":
        cmp = statusOrder(a) - statusOrder(b);
        break;
    }
    return state.sortAsc ? cmp : -cmp;
  });
}

function fmtStatus(it) {
  if (it.polished) return { text: "Polished", cls: "polished" };
  if (it.done) return { text: "OCR Done", cls: "ocr-done" };
  return { text: "Pending", cls: "pending" };
}

function renderTable() {
  const tbody = el("filesTbody");
  tbody.innerHTML = "";
  
  for (const it of state.items) {
    const tr = document.createElement("tr");
    const isCurrent = state.jobRunning && state.progress.currentFile === it.name;
    if (isCurrent) tr.classList.add("current");
    
    // Checkbox
    const tdC = document.createElement("td");
    tdC.className = "colCheck";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selection.has(it.path);
    cb.addEventListener("change", () => {
      if (cb.checked) state.selection.add(it.path);
      else state.selection.delete(it.path);
      updateStats();
      updateCheckAll();
      updateRunButton();
    });
    tdC.appendChild(cb);
    
    // File name
    const tdN = document.createElement("td");
    const nameSpan = document.createElement("span");
    nameSpan.className = "file-name";
    nameSpan.textContent = it.name;
    tdN.appendChild(nameSpan);
    
    // Size
    const tdS = document.createElement("td");
    tdS.className = "colSize";
    tdS.textContent = it.size_kb ? `${it.size_kb} KB` : "-";
    
    // Status
    const tdSt = document.createElement("td");
    tdSt.className = "colStatus";
    const st = fmtStatus(it);
    const span = document.createElement("span");
    span.className = `status ${st.cls}`;
    span.textContent = st.text;
    tdSt.appendChild(span);
    
    tr.appendChild(tdC);
    tr.appendChild(tdN);
    tr.appendChild(tdS);
    tr.appendChild(tdSt);
    tbody.appendChild(tr);
  }
  
  // Footer
  const footer = el("tableFooter");
  if (state.items.length) {
    const totalSize = state.items.reduce((s, i) => s + (i.size_kb || 0), 0);
    footer.textContent = `${state.items.length} files • ${totalSize.toLocaleString()} KB total`;
  } else {
    footer.textContent = "No markdown files found";
  }
  
  updateStats();
  updateCheckAll();
}

function updateCheckAll() {
  const checkAll = el("checkAll");
  if (state.items.length === 0) {
    checkAll.checked = false;
    checkAll.indeterminate = false;
  } else if (state.selection.size === 0) {
    checkAll.checked = false;
    checkAll.indeterminate = false;
  } else if (state.selection.size === state.items.length) {
    checkAll.checked = true;
    checkAll.indeterminate = false;
  } else {
    checkAll.checked = false;
    checkAll.indeterminate = true;
  }
}

function updateRunButton() {
  el("btnRun").disabled = !state.llmAvailable || state.selection.size === 0 || state.jobRunning;
}

// =============================================================================
// Selection
// =============================================================================

function selectAll() {
  state.selection.clear();
  for (const it of state.items) {
    if (it.done) state.selection.add(it.path);
  }
  renderTable();
  updateRunButton();
}

function selectUnpolished() {
  state.selection.clear();
  for (const it of state.items) {
    if (it.done && !it.polished) state.selection.add(it.path);
  }
  renderTable();
  updateRunButton();
}

function selectNone() {
  state.selection.clear();
  renderTable();
  updateRunButton();
}

function toggleSelectAll() {
  if (state.selection.size === state.items.length) {
    selectNone();
  } else {
    selectAll();
  }
}

// =============================================================================
// API Calls
// =============================================================================

async function loadFiles(dirOverride) {
  try {
    const url = dirOverride ? `/api/files/list?dir=${encodeURIComponent(dirOverride)}` : "/api/files/list";
    const res = await fetch(url);
    const data = await res.json();
    
    if (data.error) {
      showToast(data.error, "error");
      return;
    }
    
    state.outputDir = data.dir || "";
    if (state.outputDir) saveSessionState({ outputDir: state.outputDir });
    state.items = data.files || [];
    
    // Default: select unpolished files that are OCR'd
    state.selection.clear();
    for (const it of state.items) {
      if (it.done && !it.polished) {
        state.selection.add(it.path);
      }
    }
    
    sortItems();
    renderTable();
    
    el("fileSummary").textContent = `${state.items.length} files in ${state.outputDir}`;
    appendLog(`[ui] Loaded ${state.items.length} markdown files`);
    
  } catch (err) {
    appendLog(`[ui] Load failed: ${err.message}`);
    showToast("Failed to load files", "error");
  }
}

async function startJob() {
  if (state.selection.size === 0) {
    showToast("No files selected", "error");
    return;
  }
  
  if (!state.llmAvailable) {
    showToast("LM Studio is not running", "error");
    return;
  }
  
  el("btnRun").disabled = true;
  el("btnStop").disabled = false;
  state.jobRunning = true;
  state.progress = {
    fileIndex: 0,
    totalFiles: state.selection.size,
    currentFile: "",
    chunkIndex: 0,
    totalChunks: 0,
    speed: 0,
    eta: "",
    filesModified: 0,
  };
  updateJobStatus();
  el("progressCard").hidden = false;
  updateProgressPanel();
  
  const payload = {
    files: Array.from(state.selection),
  };
  
  try {
    const res = await fetch("/api/llm/job/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Start failed");
    }
    
    showToast(`Started LLM polish for ${state.selection.size} files`, "success");
    
  } catch (err) {
    appendLog(`[ui] Start failed: ${err.message}`);
    showToast(err.message, "error");
    el("btnRun").disabled = false;
    el("btnStop").disabled = true;
    state.jobRunning = false;
    updateJobStatus();
    el("progressCard").hidden = true;
  }
}

async function stopJob() {
  el("btnStop").disabled = true;
  
  try {
    const res = await fetch("/api/llm/job/stop", { method: "POST" });
    if (!res.ok) throw new Error("Stop failed");
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
  if (state.sse) state.sse.close();
  
  const es = new EventSource("/api/llm/job/stream");
  state.sse = es;
  
  es.addEventListener("ready", () => {
    appendLog("[ui] Connected to LLM job stream");
  });
  
  es.addEventListener("llm_log", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      appendLog(data.line);
    } catch {
      appendLog(ev.data);
    }
  });
  
  es.addEventListener("llm_file_start", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.currentFile = data.file || data.current_file || "";
      state.progress.fileIndex = data.file_index || 0;
      state.progress.totalFiles = data.total_files || state.progress.totalFiles;
      state.progress.chunkIndex = 0;
      state.progress.totalChunks = 0;
      updateJobStatus();
      updateProgressPanel();
      renderTable(); // Highlight current file
    } catch {}
  });
  
  es.addEventListener("llm_chunks_info", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.totalChunks = data.total_chunks || 0;
      updateProgressPanel();
    } catch {}
  });
  
  es.addEventListener("llm_chunk_done", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.chunkIndex = data.chunk_index || 0;
      state.progress.totalChunks = data.total_chunks || state.progress.totalChunks;
      state.progress.speed = data.speed_chars_per_sec || 0;
      state.progress.eta = data.eta || "";
      updateProgressPanel();
    } catch {}
  });
  
  es.addEventListener("llm_chunk_diff", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      updateChunkDiff(data.original || "", data.result || "");
    } catch {}
  });
  
  es.addEventListener("llm_file_done", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.progress.fileIndex = data.file_index || state.progress.fileIndex;
      state.progress.filesModified = data.files_modified || state.progress.filesModified;
      
      // Mark file as polished
      if (data.modified) {
        const file = state.items.find(i => i.name === state.progress.currentFile);
        if (file) {
          file.polished = true;
          state.selection.delete(file.path);
        }
      }
      
      updateJobStatus();
      updateProgressPanel();
      renderTable();
    } catch {}
  });
  
  es.addEventListener("llm_job_complete", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      state.jobRunning = false;
      el("btnRun").disabled = false;
      el("btnStop").disabled = true;
      updateJobStatus();
      
      if (data.success) {
        showToast(`LLM polish complete! ${data.files_modified || 0} files modified`, "success");
      } else {
        showToast("LLM polish finished with errors", "error");
      }
      
      // Refresh to get accurate status
      setTimeout(() => loadFiles(), 500);
    } catch {}
  });
  
  es.addEventListener("llm_job_stopped", () => {
    state.jobRunning = false;
    el("btnRun").disabled = false;
    el("btnStop").disabled = true;
    updateJobStatus();
  });
  
  es.addEventListener("llm_error", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      showToast(data.error || "Error occurred", "error");
    } catch {}
  });
  
  es.onerror = () => {
    // Browser will auto-reconnect
  };
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
  const transition = StateManager.consumeTransition();
  if (transition && transition.outputDir) {
    state.outputDir = transition.outputDir;
  }

  // Theme
  initTheme();
  el("btnTheme").addEventListener("click", toggleTheme);

  loadSessionState();
  
  // Selection buttons
  el("btnSelectAll").addEventListener("click", selectAll);
  el("btnSelectUnpolished").addEventListener("click", selectUnpolished);
  el("btnSelectNone").addEventListener("click", selectNone);
  el("checkAll").addEventListener("change", toggleSelectAll);
  
  // Job controls
  el("btnRun").addEventListener("click", startJob);
  el("btnStop").addEventListener("click", stopJob);
  el("btnRefresh").addEventListener("click", () => {
    checkLlmStatus();
    loadFiles(state.outputDir || undefined);
  });
  
  // Log controls
  el("btnClearLog").addEventListener("click", clearLog);
  el("autoScroll").addEventListener("change", (e) => {
    state.autoScroll = e.target.checked;
    saveSessionState({ autoScroll: state.autoScroll });
  });
  
  // Sortable headers
  $$("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (state.sortColumn === col) {
        state.sortAsc = !state.sortAsc;
      } else {
        state.sortColumn = col;
        state.sortAsc = true;
      }
      sortItems();
      renderTable();
      updateSortHeaders();
      saveSessionState({ sortColumn: state.sortColumn, sortAsc: state.sortAsc });
    });
  });
  
  // Connect SSE
  connectSSE();
  
  // Initial load
  checkLlmStatus();
  await loadFiles(state.outputDir || undefined);
  await restoreLlmJobStatus();

  if (transition) {
    if (transition.files && transition.files.length > 0) {
      state.selection.clear();
      for (const filePath of transition.files) {
        const item = state.items.find(i => i.path === filePath);
        if (item) state.selection.add(item.path);
      }
      renderTable();
      updateRunButton();
      appendLog(`[ui] Auto-selected ${state.selection.size} files from OCR`);
    }

    if (transition.autoStart && state.llmAvailable && state.selection.size > 0) {
      showToast("Auto-starting LLM polish...", "info");
      setTimeout(() => startJob(), 800);
    }
  }
  
  // Periodic LM Studio check
  setInterval(checkLlmStatus, 30000);
}

function updateSortHeaders() {
  $$("th.sortable").forEach(th => {
    th.classList.remove("asc", "desc");
    if (th.dataset.col === state.sortColumn) {
      th.classList.add(state.sortAsc ? "asc" : "desc");
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((err) => {
    console.error(err);
  });
});
