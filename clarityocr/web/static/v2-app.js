const FINAL_STATUSES = new Set(["completed", "partial", "failed", "canceled"]);
const ACTIVE_STATUSES = new Set(["running", "queued", "accepted"]);

const state = {
  apiBase: "",
  clientId: "vps-console",
  uploadId: null,
  uploadedInputs: [],
  currentJobId: null,
  currentJobStatus: null,
  pollTimer: null,
  queueTimer: null,
  allJobs: [],
};

function el(id) {
  return document.getElementById(id);
}

function normalizeApiBase(value) {
  const raw = (value || "").trim();
  if (!raw) return "";
  return raw.replace(/\/+$/, "");
}

function apiUrl(path) {
  return `${state.apiBase}${path}`;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    const detail = data?.detail || data?.error || `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return data;
}

function statusClass(status) {
  return `status ${String(status || "").toLowerCase()}`;
}

function shortPath(path) {
  if (!path) return "-";
  const chunks = String(path).split(/[\\/]/);
  return chunks[chunks.length - 1] || path;
}

function isoNow() {
  return new Date().toISOString();
}

function newRequestId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function setReadyBadge(kind, text) {
  const badge = el("readyBadge");
  badge.textContent = text;
  badge.className = "badge " + (kind === "ok" ? "badge-green" : kind === "warn" ? "badge-amber" : "badge-red");
}

function setJobBadge(text) {
  el("jobBadge").textContent = text;
}

// ─── Topbar clock ──────────────────────────────────────────────
function startClock() {
  const clockEl = el("topbarClock");
  if (!clockEl) return;
  function tick() {
    clockEl.textContent = new Date().toLocaleTimeString();
  }
  tick();
  setInterval(tick, 1000);
}

function appendEventsLog(events) {
  const lines = events.map((event) => {
    const ts = event.timestamp ? new Date(event.timestamp).toLocaleString() : isoNow();
    const filePart = event.file_id ? ` file=${shortPath(event.file_id)}` : "";
    const payloadText = event.payload ? ` ${JSON.stringify(event.payload)}` : "";
    return `[${ts}] ${event.event_code}${filePart}${payloadText}`;
  });
  el("eventsLog").textContent = lines.join("\n");
}

function renderSelectedFiles(files) {
  const list = el("selectedList");
  const count = el("selectedCount");
  list.innerHTML = "";
  if (!files.length) {
    count.textContent = "No files selected";
    return;
  }
  count.textContent = `${files.length} file(s) selected`;
  for (const file of files) {
    const li = document.createElement("li");
    li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    list.appendChild(li);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// 2.3: Queue Overview
// ──────────────────────────────────────────────────────────────────────────────

function renderQueueOverview(jobs) {
  state.allJobs = jobs || [];

  const counts = { running: 0, queued: 0, accepted: 0, completed: 0, failed: 0, partial: 0, canceled: 0 };
  for (const job of jobs) {
    const s = job.status || "unknown";
    if (counts[s] !== undefined) counts[s]++;
    else counts[s] = 1;
  }

  const runningTotal = (counts.running || 0) + (counts.accepted || 0);
  const failedTotal = (counts.failed || 0) + (counts.partial || 0);

  el("statRunning").textContent = `running: ${runningTotal}`;
  el("statRunning").className = `queue-stat${runningTotal > 0 ? " active" : ""}`;
  el("statQueued").textContent = `queued: ${counts.queued || 0}`;
  el("statQueued").className = `queue-stat${counts.queued > 0 ? " pending" : ""}`;
  el("statCompleted").textContent = `completed: ${counts.completed || 0}`;
  el("statCompleted").className = `queue-stat${counts.completed > 0 ? " ok" : ""}`;
  el("statFailed").textContent = `failed: ${failedTotal}`;
  el("statFailed").className = `queue-stat${failedTotal > 0 ? " err" : ""}`;
  el("statCanceled").textContent = `canceled: ${counts.canceled || 0}`;
  el("statCanceled").className = "queue-stat";

  el("queueLastUpdated").textContent = `updated ${new Date().toLocaleTimeString()}`;

  const container = el("activeJobsList");
  const activeJobs = jobs.filter((j) => ACTIVE_STATUSES.has(j.status));
  if (!activeJobs.length) {
    container.innerHTML = '<div class="no-active-jobs">No active jobs</div>';
    return;
  }

  container.innerHTML = activeJobs
    .map((job) => {
      const progPct = job.overall_progress_pct ?? 0;
      return `
      <div class="active-job-row">
        <button class="btn ghost btn-monitor btn-monitor-queue" data-job-id="${job.job_id}">
          <span class="mono">${job.job_id.slice(0, 8)}…</span>
        </button>
        <span class="${statusClass(job.status)}">${job.status}</span>
        <span class="active-job-mode">${job.mode || ""}</span>
        <div class="progress-bar-track mini">
          <div class="progress-bar-fill" style="width:${progPct}%"></div>
        </div>
        <span class="active-job-pct">${progPct}%</span>
      </div>`;
    })
    .join("");
}

// ──────────────────────────────────────────────────────────────────────────────
// 2.1: Progress Dashboard + Stage Timeline
// ──────────────────────────────────────────────────────────────────────────────

function updateProgressDashboard(job, files) {
  const section = el("progressSection");
  if (!job) {
    section.style.display = "none";
    return;
  }
  section.style.display = "";

  const pct = job.overall_progress_pct ?? 0;
  el("progressBarFill").style.width = `${pct}%`;
  el("progressLabel").textContent = `${pct}%`;
  el("progressJobId").textContent = job.job_id;
  el("progressFilesLabel").textContent = `${job.files_done ?? 0} / ${job.files_total ?? 0} files`;

  // ETA + elapsed time
  if (job.eta_seconds != null && job.eta_seconds > 0) {
    const mins = Math.floor(job.eta_seconds / 60);
    const secs = job.eta_seconds % 60;
    el("progressEta").textContent = `ETA: ${mins > 0 ? mins + "m " : ""}${secs}s`;
  } else if (FINAL_STATUSES.has(job.status)) {
    el("progressEta").textContent = `✓ ${job.status}`;
  } else if (job.accepted_at && !FINAL_STATUSES.has(job.status)) {
    const startTime = new Date(job.accepted_at + "Z");
    const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000);
    const eMin = Math.floor(elapsed / 60);
    const eSec = elapsed % 60;
    el("progressEta").textContent = `⏱ ${eMin > 0 ? eMin + "m " : ""}${eSec}s elapsed`;
  } else {
    el("progressEta").textContent = "";
  }

  // Stage timeline
  const timeline = el("stageTimeline");
  if (!files || !files.length) {
    timeline.innerHTML = "";
    return;
  }

  timeline.innerHTML = files
    .map((f) => {
      const stage = f.stage || "–";
      const stagePct = f.stage_progress_pct ?? 0;
      const pagesLabel =
        f.pages_total > 0
          ? `${f.pages_done ?? 0}/${f.pages_total} pages`
          : "";
      const isError =
        f.status === "failed_recoverable" ||
        f.status === "failed_final";

      return `
      <div class="stage-row${isError ? " stage-row-error" : ""}">
        <div class="stage-file-name" title="${f.input_path || ""}">${shortPath(f.input_path)}</div>
        <div class="stage-badge-wrap">
          <span class="stage-badge stage-${(stage || "").replace(/[^a-z0-9]/g, "-")}">${stage}</span>
          ${pagesLabel ? `<span class="stage-pages">${pagesLabel}</span>` : ""}
        </div>
        <div class="stage-progress-track">
          <div class="stage-progress-fill" style="width:${stagePct}%"></div>
        </div>
        <span class="${statusClass(f.status)}">${f.status}</span>
      </div>`;
    })
    .join("");
}

// ──────────────────────────────────────────────────────────────────────────────
// 2.2: Enhanced files table with error display + retry per file
// ──────────────────────────────────────────────────────────────────────────────

function renderJobsTable(jobs) {
  const tbody = el("jobsTbody");
  tbody.innerHTML = "";
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="5">No jobs yet</td></tr>';
    return;
  }
  for (const job of jobs) {
    const pct = job.overall_progress_pct;
    const progressCell = (pct != null)
      ? `<div class="progress-bar-track mini"><div class="progress-bar-fill" style="width:${pct}%"></div></div> <span class="mono" style="font-size:0.75rem">${pct}%</span>`
      : `–`;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${job.job_id}</td>
      <td><span class="${statusClass(job.status)}">${job.status}</span></td>
      <td>${job.mode}</td>
      <td>${progressCell}</td>
      <td><button class="btn ghost btn-monitor" data-job-id="${job.job_id}">Monitor</button></td>
    `;
    tbody.appendChild(tr);
  }
}

function renderFilesTable(files) {
  const tbody = el("filesTbody");
  tbody.innerHTML = "";
  if (!files.length) {
    tbody.innerHTML = '<tr><td colspan="6">No files tracked yet</td></tr>';
    el("filesSummary").textContent = "-";
    return;
  }

  const counts = {};
  for (const file of files) {
    counts[file.status] = (counts[file.status] || 0) + 1;

    const isError =
      file.status === "failed_recoverable" ||
      file.status === "failed_final";
    const stage = file.stage || "–";
    const stagePct = file.stage_progress_pct ?? 0;

    const errorCell = isError && file.error
      ? `<span class="file-error-msg" title="${file.error}">${String(file.error).slice(0, 60)}${file.error.length > 60 ? "…" : ""}</span>`
      : (file.error ? `<span class="mono" style="font-size:0.75rem" title="${file.error}">${String(file.error).slice(0, 40)}${file.error.length > 40 ? "…" : ""}</span>` : "–");

    const stageCell = `
      <span class="stage-badge stage-${(stage).replace(/[^a-z0-9]/g, "-")}">${stage}</span>
      ${stagePct > 0 ? `<div class="stage-progress-track mini"><div class="stage-progress-fill" style="width:${stagePct}%"></div></div>` : ""}`;

    // Per-file retry: only show if has failures and has retries remaining
    const canRetry = isError && file.attempt < file.max_attempts;
    const retryBtn = canRetry
      ? `<button class="btn danger btn-retry-file" data-job-id="${state.currentJobId}" style="padding:4px 8px; font-size:0.75rem">Retry</button>`
      : "";

    const tr = document.createElement("tr");
    if (isError) tr.classList.add("file-row-error");
    tr.innerHTML = `
      <td title="${file.input_path}">${shortPath(file.input_path)}</td>
      <td>${stageCell}</td>
      <td><span class="${statusClass(file.status)}">${file.status}</span></td>
      <td>${file.attempt}/${file.max_attempts}</td>
      <td>${errorCell}</td>
      <td>${retryBtn}</td>
    `;
    tbody.appendChild(tr);
  }

  const summary = Object.entries(counts)
    .map(([key, value]) => `${key}:${value}`)
    .join(" | ");
  el("filesSummary").textContent = summary;
}

function renderArtifacts(artifacts) {
  const list = el("artifactsList");
  list.innerHTML = "";
  if (!artifacts.length) {
    list.innerHTML = "<li>No artifacts yet</li>";
    return;
  }
  for (const artifact of artifacts) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span><strong>${artifact.type}</strong> - ${shortPath(artifact.path)}</span>
      <a class="download-link" href="${apiUrl(artifact.download_url)}" target="_blank" rel="noopener">download</a>
    `;
    list.appendChild(li);
  }
}

function saveUiState() {
  localStorage.setItem("v2-api-base", state.apiBase);
  localStorage.setItem("v2-client-id", state.clientId);
  if (state.currentJobId) {
    localStorage.setItem("v2-current-job-id", state.currentJobId);
  } else {
    localStorage.removeItem("v2-current-job-id");
  }
}

function loadUiState() {
  state.apiBase = normalizeApiBase(localStorage.getItem("v2-api-base") || "");
  state.clientId = localStorage.getItem("v2-client-id") || "vps-console";
  state.currentJobId = localStorage.getItem("v2-current-job-id");
  el("apiBase").value = state.apiBase;
  el("clientId").value = state.clientId;
  el("currentJob").textContent = state.currentJobId || "-";
}

async function refreshHealth() {
  try {
    const ready = await apiFetch("/api/v2/health/ready");
    el("healthText").textContent = `ocr=${ready.ocr_core}, db=${ready.db}, llm=${ready.llm}`;
    setReadyBadge(ready.llm === "ready" ? "ok" : "warn", `ready (${ready.llm})`);
  } catch (err) {
    el("healthText").textContent = String(err.message || err);
    setReadyBadge("err", "offline");
  }
}

async function refreshJobs() {
  try {
    const data = await apiFetch("/api/v2/jobs?limit=20");
    renderJobsTable(data.jobs || []);
    // Also refresh queue overview with same data
    renderQueueOverview(data.jobs || []);
  } catch (err) {
    renderJobsTable([]);
    renderQueueOverview([]);
    setJobBadge(`jobs fetch failed: ${err.message}`);
  }
}

function updateJobControls(status) {
  const hasJob = Boolean(state.currentJobId);
  const cancelEnabled = hasJob && !FINAL_STATUSES.has(status || "");
  const retryEnabled = hasJob && (status === "failed" || status === "partial" || status === "canceled");
  el("btnCancel").disabled = !cancelEnabled;
  el("btnRetry").disabled = !retryEnabled;
}

async function refreshCurrentJob() {
  if (!state.currentJobId) {
    updateProgressDashboard(null, null);
    return;
  }
  try {
    const [job, filesData, events, artifacts] = await Promise.all([
      apiFetch(`/api/v2/jobs/${state.currentJobId}`),
      apiFetch(`/api/v2/jobs/${state.currentJobId}/files`),
      apiFetch(`/api/v2/jobs/${state.currentJobId}/events?limit=300`),
      apiFetch(`/api/v2/jobs/${state.currentJobId}/artifacts`),
    ]);

    state.currentJobStatus = job.status;
    el("currentJob").textContent = state.currentJobId;
    setJobBadge(`${job.status} | ${job.mode}`);
    updateJobControls(job.status);

    const files = filesData.files || [];
    renderFilesTable(files);
    appendEventsLog(events.events || []);
    renderArtifacts(artifacts.artifacts || []);

    // 2.1: Progress dashboard
    updateProgressDashboard(job, files);

    if (FINAL_STATUSES.has(job.status)) {
      stopPolling();
      setJobBadge(`final: ${job.status}`);
      refreshJobs();
    }
  } catch (err) {
    setJobBadge(`job error: ${err.message}`);
  }
}

function startPolling() {
  stopPolling();
  // Poll every 3 seconds for active jobs
  state.pollTimer = setInterval(refreshCurrentJob, 3000);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function startQueuePolling() {
  stopQueuePolling();
  state.queueTimer = setInterval(refreshJobs, 5000);
}

function stopQueuePolling() {
  if (state.queueTimer) {
    clearInterval(state.queueTimer);
    state.queueTimer = null;
  }
}

async function uploadFiles() {
  const files = Array.from(el("fileInput").files || []);
  if (!files.length) {
    setJobBadge("choose files first");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  // Use XMLHttpRequest for upload progress tracking
  setJobBadge("uploading... 0%");
  try {
    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          setJobBadge(`uploading... ${pct}%`);
        }
      });
      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); }
          catch { resolve({}); }
        } else {
          let detail = `${xhr.status} ${xhr.statusText}`;
          try { const d = JSON.parse(xhr.responseText); detail = d.detail || detail; } catch {}
          reject(new Error(detail));
        }
      });
      xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
      xhr.addEventListener("abort", () => reject(new Error("Upload aborted")));
      xhr.open("POST", apiUrl("/api/v2/uploads"));
      xhr.send(formData);
    });
    state.uploadId = data.upload_id;
    state.uploadedInputs = data.inputs || [];
    el("uploadInfo").textContent = `Upload batch: ${state.uploadId} | server files: ${state.uploadedInputs.length}`;
    setJobBadge("upload complete ✓");
  } catch (err) {
    setJobBadge(`upload failed: ${err.message}`);
    el("jobBadge")?.classList.add("badge-error");
    setTimeout(() => el("jobBadge")?.classList.remove("badge-error"), 5000);
  }
}

async function startJob() {
  if (!state.uploadedInputs.length) {
    setJobBadge("upload files before starting");
    return;
  }

  const payload = {
    api_version: "v2",
    meta_schema_version: "1.0",
    client_id: (el("clientId").value || "vps-console").trim(),
    client_request_id: newRequestId(),
    inputs: state.uploadedInputs,
    mode: el("mode").value,
    preset: el("preset").value,
    naming_policy: el("namingPolicy").value,
    polish: el("polish").value,
  };

  try {
    setJobBadge("submitting...");
    const data = await apiFetch("/api/v2/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.currentJobId = data.job_id;
    state.currentJobStatus = data.status;
    saveUiState();
    el("currentJob").textContent = data.job_id;
    setJobBadge(`queued: ${data.job_id}`);
    startPolling();
    refreshCurrentJob();
    refreshJobs();
  } catch (err) {
    setJobBadge(`start failed: ${err.message}`);
  }
}

async function cancelJob() {
  if (!state.currentJobId) return;
  try {
    const data = await apiFetch(`/api/v2/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    state.currentJobStatus = data.status;
    setJobBadge(`cancel: ${data.status}`);
    refreshCurrentJob();
  } catch (err) {
    setJobBadge(`cancel failed: ${err.message}`);
  }
}

async function retryJob() {
  if (!state.currentJobId) return;
  try {
    const data = await apiFetch(`/api/v2/jobs/${state.currentJobId}/retry-failed`, { method: "POST" });
    setJobBadge(`requeued: ${data.requeued_count}`);
    startPolling();
    refreshCurrentJob();
  } catch (err) {
    setJobBadge(`retry failed: ${err.message}`);
  }
}

async function retryFileJob(jobId) {
  try {
    const data = await apiFetch(`/api/v2/jobs/${jobId}/retry-failed`, { method: "POST" });
    setJobBadge(`requeued: ${data.requeued_count}`);
    startPolling();
    if (jobId === state.currentJobId) refreshCurrentJob();
  } catch (err) {
    setJobBadge(`retry failed: ${err.message}`);
  }
}

function bindEvents() {
  el("btnSaveApiBase").addEventListener("click", () => {
    state.apiBase = normalizeApiBase(el("apiBase").value);
    state.clientId = (el("clientId").value || "vps-console").trim() || "vps-console";
    saveUiState();
    refreshHealth();
    refreshJobs();
    refreshCurrentJob();
  });

  el("fileInput").addEventListener("change", (event) => {
    const files = Array.from(event.target.files || []);
    renderSelectedFiles(files);
  });

  el("btnUpload").addEventListener("click", uploadFiles);
  el("btnStart").addEventListener("click", startJob);
  el("btnCancel").addEventListener("click", cancelJob);
  el("btnRetry").addEventListener("click", retryJob);
  el("btnRefreshNow").addEventListener("click", async () => {
    await refreshJobs();
    await refreshCurrentJob();
  });
  el("btnReloadJobs").addEventListener("click", refreshJobs);
  el("btnReloadQueue").addEventListener("click", refreshJobs);

  // Monitor button in jobs table
  el("jobsTbody").addEventListener("click", (event) => {
    const button = event.target.closest(".btn-monitor");
    if (!button) return;
    const jobId = button.getAttribute("data-job-id");
    if (!jobId) return;
    state.currentJobId = jobId;
    saveUiState();
    el("currentJob").textContent = jobId;
    setJobBadge(`monitoring: ${jobId}`);
    startPolling();
    refreshCurrentJob();
  });

  // Monitor button in queue overview
  el("activeJobsList").addEventListener("click", (event) => {
    const button = event.target.closest(".btn-monitor-queue");
    if (!button) return;
    const jobId = button.getAttribute("data-job-id");
    if (!jobId) return;
    state.currentJobId = jobId;
    saveUiState();
    el("currentJob").textContent = jobId;
    setJobBadge(`monitoring: ${jobId}`);
    startPolling();
    refreshCurrentJob();
    window.scrollTo({ top: el("progressSection").offsetTop - 20, behavior: "smooth" });
  });

  // Per-file retry button (2.2)
  el("filesTbody").addEventListener("click", (event) => {
    const button = event.target.closest(".btn-retry-file");
    if (!button) return;
    const jobId = button.getAttribute("data-job-id");
    if (!jobId) return;
    retryFileJob(jobId);
  });
}

async function init() {
  loadUiState();
  bindEvents();
  startClock();
  updateJobControls(null);
  renderSelectedFiles([]);
  renderFilesTable([]);
  renderArtifacts([]);
  appendEventsLog([]);
  updateProgressDashboard(null, null);
  await refreshHealth();
  await refreshJobs();
  if (state.currentJobId) {
    startPolling();
    await refreshCurrentJob();
  }
  // 2.3: Start queue auto-refresh (every 5 seconds)
  startQueuePolling();
}

window.addEventListener("beforeunload", () => {
  saveUiState();
  stopPolling();
  stopQueuePolling();
});

window.addEventListener("DOMContentLoaded", () => {
  init().catch((err) => {
    setJobBadge(`init failed: ${err.message}`);
  });
});
