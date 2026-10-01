"use strict";
// Transcription app: #/ upload, #/progress/<id>, #/edit/<id>[/<chunk>]

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});
const fmt = (s) => {
  s = Math.max(0, s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(sec).padStart(2, "0");
};
let toastTimer;
function toast(msg, ms = 3500) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

// ------------------------------------------------------------------ routing
let poller = null;
function show(view) {
  for (const v of ["viewUpload", "viewProgress", "viewEdit"]) $(v).classList.toggle("hidden", v !== view);
}
function route() {
  clearInterval(poller); poller = null;
  const [, page, id, chunk] = (location.hash || "#/").split("/");
  $("exportMenu").classList.add("hidden"); $("headerStats").textContent = ""; $("crumb").textContent = "";
  if (page === "progress" && id) return openProgress(id);
  if (page === "edit" && id) return openEditor(id, chunk ? parseInt(chunk, 10) : null);
  openUpload();
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ 1. load audio
let chosenFile = null;
function openUpload() {
  show("viewUpload");
  api("/api/sessions").then((list) => {
    const ul = $("sessions"); ul.innerHTML = "";
    if (!list.length) { ul.innerHTML = '<li class="muted">None yet.</li>'; return; }
    for (const s of list) {
      const li = document.createElement("li");
      const target = s.status === "ready" ? `#/edit/${s.id}` : `#/progress/${s.id}`;
      li.innerHTML = `<a href="${target}"></a><span class="muted"></span><span class="muted"></span>`;
      li.children[0].textContent = s.title || s.id;
      li.children[1].textContent = s.status === "ready" ? `${s.edited} of ${s.chunks} chunks corrected` : s.status;
      li.children[2].textContent = (s.duration ? fmt(s.duration) + " · " : "") + (s.created_at || "").replace("T", " ").slice(0, 16);
      ul.appendChild(li);
    }
  }).catch(() => {});
}
function choose(file) {
  if (!file) return;
  chosenFile = file;
  $("chosenName").textContent = file.name;
  $("chosenSize").textContent = (file.size / 1048576).toFixed(1) + " MB";
  $("preview").src = URL.createObjectURL(file);
  $("chosen").classList.remove("hidden");
}
$("file").addEventListener("change", (e) => choose(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); choose(e.dataTransfer.files[0]); });

$("transcribe").addEventListener("click", () => {
  if (!chosenFile) return;
  const form = new FormData();
  form.append("file", chosenFile);
  form.append("language", $("language").value);
  form.append("model", $("model").value);
  form.append("speakers", $("speakers").checked ? "true" : "false");
  const xhr = new XMLHttpRequest();
  const bar = $("uploadBar"); bar.classList.remove("hidden");
  $("transcribe").disabled = true; $("transcribe").textContent = "Uploading…";
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.firstElementChild.style.width = (100 * e.loaded / e.total) + "%"; };
  xhr.onload = () => {
    $("transcribe").disabled = false; $("transcribe").textContent = "Transcribe"; bar.classList.add("hidden");
    if (xhr.status !== 200) return toast("Upload failed: " + xhr.responseText);
    $("preview").pause();
    location.hash = "#/progress/" + JSON.parse(xhr.responseText).id;
  };
  xhr.onerror = () => { $("transcribe").disabled = false; $("transcribe").textContent = "Transcribe"; toast("Upload failed."); };
  xhr.open("POST", "/api/sessions"); xhr.send(form);
});

// ------------------------------------------------------------------ 2. progress
function openProgress(id) {
  show("viewProgress");
  $("failure").classList.add("hidden");
  const started = Date.now(); let firstDone = null;
  const tick = () => api(`/api/sessions/${id}`).then((s) => {
    $("crumb").textContent = s.recording_id || "";
    if (s.status === "ready") { location.hash = `#/edit/${id}`; return; }
    if (s.status === "failed") {
      clearInterval(poller);
      $("failure").textContent = "Transcription failed: " + (s.error || "unknown error");
      $("failure").classList.remove("hidden"); return;
    }
    const p = s.progress || {};
    $("stage").textContent = p.total ? `${p.stage}: chunk ${p.done + 1} of ${p.total}` : (p.stage || "Waiting to start…");
    $("progressFill").style.width = p.total ? (100 * p.done / p.total) + "%" : "4%";
    if (p.total && p.done) {
      if (firstDone === null) firstDone = { t: Date.now(), d: p.done };
      const rate = (Date.now() - firstDone.t) / Math.max(1, p.done - firstDone.d);
      if (p.done > firstDone.d) $("eta").textContent = "About " + fmt(rate * (p.total - p.done) / 1000) + " left";
    } else {
      $("eta").textContent = "Elapsed " + fmt((Date.now() - started) / 1000);
    }
  }).catch((e) => toast(e.message));
  tick(); poller = setInterval(tick, 2000);
}

// ------------------------------------------------------------------ 3. correct
const E = { id: null, session: null, index: 0, loadedText: "", dirty: false };
const audio = $("audio");

function openEditor(id, chunk) {
  show("viewEdit");
  api(`/api/sessions/${id}`).then((s) => {
    if (s.status !== "ready") { location.hash = `#/progress/${id}`; return; }
    E.id = id; E.session = s;
    $("crumb").textContent = s.recording_id || "";
    for (const f of ["Txt", "Srt", "Json"]) $("export" + f).href = `/api/sessions/${id}/export?format=${f.toLowerCase()}`;
    $("exportMenu").classList.remove("hidden");
    renderList();
    const firstOpen = s.segments.findIndex((x) => !x.edited);
    load(chunk !== null && !isNaN(chunk) ? chunk : Math.max(0, firstOpen), false);
    poller = setInterval(refresh, 4000);
  }).catch((e) => toast(e.message));
}

function stats() {
  const segs = E.session.segments, done = segs.filter((x) => x.edited).length;
  $("headerStats").textContent = `${done} of ${segs.length} corrected`;
}
function renderList() {
  const ol = $("chunkList"); ol.innerHTML = "";
  E.session.segments.forEach((seg, i) => {
    const li = document.createElement("li");
    li.dataset.i = i;
    li.innerHTML = '<span class="mark"></span><span class="mono"></span><span class="snippet"></span>';
    li.onclick = () => go(i, true);
    ol.appendChild(li); paintItem(i);
  });
  stats();
}
function paintItem(i) {
  const li = $("chunkList").children[i]; if (!li) return;
  const seg = E.session.segments[i];
  li.className = (i === E.index ? "current " : "") + (seg.edited ? "edited" : (seg.suggestion === "lookahead" || seg.suggestion === "adapted") ? "updated" : "");
  li.children[0].textContent = seg.edited ? "✓" : (seg.suggestion === "lookahead" || seg.suggestion === "adapted") ? "↻" : "·";
  li.children[1].textContent = String(i + 1).padStart(3, " ") + " " + fmt(seg.start);
  li.children[2].textContent = seg.transcript || "…";
}

function load(i, autoplay) {
  const segs = E.session.segments;
  if (!segs.length) { $("chunkTitle").textContent = "No speech found in this audio."; return; }
  i = Math.max(0, Math.min(segs.length - 1, i));
  const prev = E.index; E.index = i;
  paintItem(prev); paintItem(i);
  const seg = segs[i];
  $("chunkTitle").textContent = `Chunk ${i + 1} of ${segs.length}`;
  $("chunkTime").textContent = `${fmt(seg.start)} – ${fmt(seg.end)}`;
  $("chunkSpeaker").textContent = seg.speaker_id || ""; $("chunkSpeaker").classList.toggle("hidden", !seg.speaker_id);
  const st = $("chunkState");
  st.className = "tag " + (seg.edited ? "edited" : seg.suggestion === "transcript" ? "" : seg.suggestion !== "model" ? "updated" : "");
  st.textContent = seg.edited ? "corrected" : seg.suggestion === "lookahead" ? "re-transcribed with your corrections"
    : seg.suggestion === "adapted" ? "your corrections applied" : seg.suggestion === "transcript" ? "aligned transcript" : "model";
  st.classList.toggle("hidden", seg.suggestion === "model" && !seg.edited);
  $("text").value = seg.transcript || ""; E.loadedText = $("text").value; E.dirty = false;
  $("modelText").textContent = seg.asr_hypothesis || "(none)";
  $("saveState").textContent = "";
  const rate = audio.playbackRate;
  audio.src = `/api/sessions/${E.id}/chunks/${i}/audio`;
  audio.playbackRate = rate;
  if (autoplay) audio.play().catch(() => {});
  history.replaceState(null, "", `#/edit/${E.id}/${i}`);
  const li = $("chunkList").children[i]; if (li) li.scrollIntoView({ block: "nearest" });
  $("text").focus();
  if (i + 1 < segs.length) new Audio(`/api/sessions/${E.id}/chunks/${i + 1}/audio`).preload = "auto";
}

async function save(force) {
  const text = $("text").value;
  if (!force && text === E.loadedText) return;
  const i = E.index;
  $("saveState").textContent = "Saving…";
  try {
    const r = await api(`/api/sessions/${E.id}/chunks/${i}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const seg = E.session.segments[i];
    Object.assign(seg, { transcript: text.trim(), edited: true, status: "approved" });
    if (E.index === i) { E.loadedText = text; $("saveState").textContent = "Saved"; }
    paintItem(i); stats();
    const rules = r.adaptation.rules || [];
    const before = (E.session.adaptation && E.session.adaptation.rules) || [];
    E.session.adaptation = r.adaptation;
    if (r.adapted.length) await refresh();
    const fresh = rules.filter((x) => !before.some((y) => y[0] === x[0] && y[1] === x[1]));
    const box = $("learned");
    if (fresh.length || r.lookahead.length) {
      box.innerHTML = "";
      if (fresh.length) box.append("Learned: " + fresh.map((x) => `${x[0]} → ${x[1]}`).join(", ") +
        (r.adapted.length ? ` (applied to ${r.adapted.length} chunks). ` : ". "));
      if (r.lookahead.length) box.append(`Re-transcribing the next ${r.lookahead.length} chunks with your corrections.`);
      box.classList.remove("hidden");
    }
  } catch (e) { $("saveState").textContent = "Not saved: " + e.message; throw e; }
}
async function go(i, autoplay) {
  if (i === E.index) return;
  try { await save(false); } catch (e) { return; }
  load(i, autoplay);
}
async function saveAndNext() {
  try { await save(true); } catch (e) { return; }
  if (E.index + 1 < E.session.segments.length) load(E.index + 1, true);
  else toast("Last chunk saved. Export from the top right.");
}

async function refresh() {          // pick up chunks improved in the background
  if (!E.id) return;
  const s = await api(`/api/sessions/${E.id}`).catch(() => null);
  if (!s || !E.session) return;
  s.segments.forEach((seg, i) => {
    const mine = E.session.segments[i];
    if (!mine || mine.edited) return;
    if (seg.transcript !== mine.transcript || seg.suggestion !== mine.suggestion) {
      E.session.segments[i] = seg; paintItem(i);
      if (i === E.index && $("text").value === E.loadedText) {          // untouched: show the better text
        $("text").value = seg.transcript; E.loadedText = seg.transcript;
        $("chunkState").textContent = "re-transcribed with your corrections";
        $("chunkState").className = "tag updated";
      }
    }
  });
  E.session.adaptation = s.adaptation;
}

// player
function setRate(r) { audio.playbackRate = Math.round(Math.min(2, Math.max(0.5, r)) * 100) / 100; $("speed").textContent = audio.playbackRate.toFixed(2) + "×"; }
function togglePlay() { if (audio.paused) audio.play().catch(() => {}); else audio.pause(); }
$("playBtn").onclick = togglePlay;
audio.onplay = () => { $("playBtn").textContent = "❚❚"; };
audio.onpause = () => { $("playBtn").textContent = "▶"; };
audio.ontimeupdate = () => {
  const d = audio.duration || 0;
  $("seek").value = d ? Math.round(1000 * audio.currentTime / d) : 0;
  $("clock").textContent = `${fmt(audio.currentTime)} / ${fmt(d)}`;
};
audio.onratechange = () => { $("speed").textContent = audio.playbackRate.toFixed(2) + "×"; };
$("seek").oninput = (e) => { if (audio.duration) audio.currentTime = audio.duration * e.target.value / 1000; };
$("prevBtn").onclick = () => go(E.index - 1, true);
$("nextBtn").onclick = saveAndNext;
$("text").addEventListener("input", () => { E.dirty = $("text").value !== E.loadedText; $("saveState").textContent = E.dirty ? "Edited" : ""; });

// keyboard: Esc play/pause, F1 back, F2 forward, F3 slower, F4 faster
document.addEventListener("keydown", (e) => {
  if ($("viewEdit").classList.contains("hidden")) return;
  const k = e.key;
  if (k === "Escape") { e.preventDefault(); togglePlay(); }
  else if (k === "F1") { e.preventDefault(); audio.currentTime = Math.max(0, audio.currentTime - 3); }
  else if (k === "F2") { e.preventDefault(); audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 3); }
  else if (k === "F3") { e.preventDefault(); setRate(audio.playbackRate - 0.1); }
  else if (k === "F4") { e.preventDefault(); setRate(audio.playbackRate + 0.1); }
  else if (k === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveAndNext(); }
  else if (e.altKey && k === "ArrowDown") { e.preventDefault(); go(E.index + 1, true); }
  else if (e.altKey && k === "ArrowUp") { e.preventDefault(); go(E.index - 1, true); }
  else if (k === "s" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save(true); }
});
// Some browsers open help on F1 even when keydown is cancelled.
window.addEventListener("help", (e) => e.preventDefault());
window.addEventListener("beforeunload", (e) => { if (E.dirty) { e.preventDefault(); e.returnValue = ""; } });

route();
