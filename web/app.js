/* DeedLens frontend: plain JavaScript, no build step.
 *
 * Safety: answers are written by an LLM that read untrusted documents, so API text is NEVER
 * inserted with innerHTML. Everything goes through textContent / createTextNode via el().
 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);

const EXAMPLES = [
  "Who owns this property now?",
  "How much did the current owners pay for it?",
  "List every owner in order, with what each one paid.",
  "How large is the lot, and what is its address?",
  "Was this property ever in a tax foreclosure?",
];

const state = {
  parcel: null,
  properties: [],
  chain: [],            // chain entries of the selected parcel (doc_id, title, num_pages, ...)
  busy: false,
  gemini: true,
  pending: null,        // the needs_review result being reviewed
  question: "",
  viewer: null,         // {docId, page, pages, title}
};

/* ---------------------------------------------------------------- helpers */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const init = { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } };
  let response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    throw new Error("Can't reach the DeedLens server. Is it running?");
  }
  let body = null;
  try { body = await response.json(); } catch (_) { /* non-JSON body */ }
  if (!response.ok) throw new Error((body && body.error) || `Request failed (${response.status})`);
  return body;
}

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const money = (v) => (v === null || v === undefined ? null : usd.format(v));
const titleCase = (s) => (s || "").replace(/\b\w/g, (c) => c.toUpperCase());
const names = (list) => (list && list.length ? list.join("; ") : "—");

function store(key, value) {
  try { localStorage.setItem(key, value); } catch (_) { /* storage may be blocked */ }
}
function recall(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}

function showError(message) {
  const box = $("#error");
  box.textContent = message || "";
  box.hidden = !message;
}

/* ---------------------------------------------------------------- status */

async function loadHealth() {
  const status = $("#status");
  const text = $(".status-text", status);
  try {
    const h = await api("/api/health");
    state.gemini = h.gemini;
    if (!h.gemini) {
      status.className = "status warn";
      text.textContent = "Search only: GEMINI_API_KEY is missing, so questions can't be answered.";
    } else if (!h.documents) {
      status.className = "status warn";
      text.textContent = "No documents ingested yet. Run the ingestion first.";
    } else {
      status.className = "status ok";
      text.textContent = `Ready · ${h.documents} documents · ${h.chunks} passages`;
    }
  } catch (err) {
    status.className = "status down";
    text.textContent = "Server unreachable";
  }
  updateControls();
}

/* ---------------------------------------------------------------- properties + chain */

function propertyLabel(p) {
  const parts = [p.parcel];
  if (p.address) parts.push(p.address.replace(/, New York$/, ""));
  if (p.current_owner) parts.push(p.current_owner);
  return `${parts.join(" · ")} (${p.documents} doc${p.documents === 1 ? "" : "s"})`;
}

async function loadProperties() {
  const select = $("#parcel");
  try {
    state.properties = await api("/api/properties");
  } catch (err) {
    select.replaceChildren(el("option", { value: "" }, "Couldn't load properties"));
    showError(err.message);
    return;
  }
  if (!state.properties.length) {
    select.replaceChildren(el("option", { value: "" }, "No properties yet"));
    return;
  }
  select.replaceChildren(
    el("option", { value: "" }, "Select a property…"),
    ...state.properties.map((p) => el("option", { value: p.parcel }, propertyLabel(p))),
  );
  select.disabled = false;
  const remembered = recall("deedlens.parcel");
  const initial = state.properties.some((p) => p.parcel === remembered) ? remembered : state.properties[0].parcel;
  select.value = initial;
  await selectParcel(initial);
}

async function selectParcel(parcel) {
  state.parcel = parcel || null;
  state.chain = [];
  resetResult();
  const card = $("#property-card");
  const chainList = $("#chain");
  chainList.replaceChildren();
  if (!parcel) {
    card.hidden = true;
    updateControls();
    return;
  }
  store("deedlens.parcel", parcel);
  const p = state.properties.find((x) => x.parcel === parcel);
  $('[data-field="address"]', card).textContent = (p && p.address) || "Not on record";
  $('[data-field="owner"]', card).textContent = (p && p.current_owner) || "Not on record";
  $('[data-field="documents"]', card).textContent = p ? String(p.documents) : "—";
  card.hidden = false;
  updateControls();

  try {
    state.chain = await api(`/api/properties/${encodeURIComponent(parcel)}/chain`);
  } catch (err) {
    showError(err.message);
    return;
  }
  if (state.parcel !== parcel) return; // the user switched again meanwhile
  chainList.replaceChildren(...state.chain.map(chainItem));
  if (!state.chain.length) chainList.append(el("li", {}, "No documents for this parcel."));
}

function chainItem(entry) {
  const priceBits = [];
  if (entry.price_reported !== null && entry.price_reported !== undefined) {
    priceBits.push(el("strong", {}, money(entry.price_reported)), " reported to the county");
    if (entry.price_in_deed !== null && entry.price_in_deed !== undefined) {
      priceBits.push(" · deed states ", money(entry.price_in_deed));
    }
  } else if (entry.price_in_deed !== null && entry.price_in_deed !== undefined) {
    priceBits.push("Deed states ", el("strong", {}, money(entry.price_in_deed)));
  }
  if (entry.nominal) priceBits.push(" ", el("span", { class: "tag", title: "A token amount, not the real price" }, "nominal"));
  if (entry.transfer_tax) priceBits.push(` · transfer tax ${money(entry.transfer_tax)}`);

  return el("li", {},
    el("div", { class: "tl-head" },
      el("span", { class: "tl-year" }, entry.year || "—"),
      el("span", { class: "tl-type" }, entry.type || "document")),
    (entry.sellers.length || entry.buyers.length)
      ? el("p", { class: "tl-parties" }, names(entry.sellers), el("span", { class: "arrow", "aria-label": "to" }, "→"),
        names(entry.buyers))
      : null,
    priceBits.length ? el("p", { class: "tl-price" }, ...priceBits) : null,
    el("button", {
      type: "button", class: "link-button",
      onclick: () => openViewer(entry.doc_id, 1, entry.title),
    }, entry.title));
}

/* ---------------------------------------------------------------- asking */

function updateControls() {
  const disabled = state.busy || !state.parcel || !state.gemini;
  $("#ask-button").disabled = disabled;
  $("#question").disabled = state.busy || !state.parcel;
  document.querySelectorAll(".example").forEach((b) => { b.disabled = disabled; });
  document.querySelectorAll(".review-actions .button").forEach((b) => { b.disabled = state.busy; });
  $("#ask-button").textContent = state.busy ? "Working…" : "Ask";
}

function setBusy(busy) {
  state.busy = busy;
  const result = $("#result");
  result.setAttribute("aria-busy", String(busy));
  if (busy) {
    result.hidden = false;
    $("#loading").hidden = false;
  } else {
    $("#loading").hidden = true;
  }
  updateControls();
}

function resetResult() {
  state.pending = null;
  $("#result").hidden = true;
  $("#answer").hidden = true;
  $("#review").hidden = true;
}

async function ask(question) {
  question = question.trim();
  if (!question || !state.parcel || state.busy) return;
  showError("");
  state.question = question;
  resetResult();
  setBusy(true);
  try {
    const result = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({ parcel: state.parcel, question }),
    });
    showResult(result);
  } catch (err) {
    showError(err.message);
    $("#result").hidden = true;
  } finally {
    setBusy(false);
  }
}

async function submitReview(action) {
  if (!state.pending || state.busy) return;
  const body = { action };
  if (action === "edit") {
    const wrap = $(".edit-wrap");
    if (wrap.hidden) { // first click on Edit: show the editor, prefilled with the draft
      wrap.hidden = false;
      const box = $("#edit-text");
      box.value = state.pending.draft;
      box.focus();
      $('[data-action="edit"]').textContent = "Submit edit";
      return;
    }
    body.text = $("#edit-text").value.trim();
    if (!body.text) { showError("Write the corrected answer first."); return; }
  }
  showError("");
  setBusy(true);
  $("#review").hidden = true;
  try {
    const result = await api(`/api/review/${encodeURIComponent(state.pending.thread_id)}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    showResult(result);
  } catch (err) {
    showError(err.message);
    $("#review").hidden = false;
  } finally {
    setBusy(false);
  }
}

/* ---------------------------------------------------------------- rendering answers */

const CITATION = /\[([^[\]]+)\]/g;
const squash = (s) => s.replace(/\s+/g, "").toLowerCase();

function findCitation(label, citations) {
  const key = squash(label);
  return (citations || []).find((c) => squash(c.label) === key) || null;
}

function inlineText(text) {
  // Minimal markdown: **bold** and "* " bullets; everything else stays plain text.
  const out = [];
  const normalized = text.replace(/^(\s*)[*-] /gm, "$1• ");
  normalized.split(/\*\*(.+?)\*\*/g).forEach((part, i) => {
    if (part) out.push(i % 2 ? el("strong", {}, part) : document.createTextNode(part));
  });
  return out;
}

function chip(citation, label) {
  const isRecord = citation.kind === "records";
  const text = isRecord ? label.replace(/^records:\s*/i, "") : label;
  return el("button", {
    type: "button", class: "chip",
    title: `Open ${citation.doc_id}, page ${citation.page}`,
    "aria-label": `Source: ${label}. Open page ${citation.page}`,
    onclick: () => openViewer(citation.doc_id, citation.page, docTitle(citation.doc_id, text)),
  }, el("span", { class: "chip-icon", "aria-hidden": "true" }, isRecord ? "▤" : "▣"), text);
}

function renderCited(text, citations) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    frag.append(...inlineText(text.slice(last, match.index)));
    const parts = match[1].split(";").map((s) => s.trim()).filter(Boolean);
    const resolved = parts.map((p) => [p, findCitation(p, citations)]);
    if (resolved.every(([, c]) => c)) {
      resolved.forEach(([label, c]) => frag.append(chip(c, label)));
    } else {
      frag.append(document.createTextNode(match[0])); // unknown source: show as written
    }
    last = match.index + match[0].length;
  }
  frag.append(...inlineText(text.slice(last)));
  return frag;
}

function pathText(result) {
  const path = (result.path || []).join(" → ");
  const route = result.route ? `route: ${result.route}` : "";
  return [path, route].filter(Boolean).join("   ·   ");
}

function showResult(result) {
  $("#result").hidden = false;
  if (result.status === "needs_review") {
    showReview(result);
    return;
  }
  state.pending = null;
  const box = $("#answer");
  $(".question-echo", box).textContent = `“${state.question}”`;
  $(".answer-text", box).replaceChildren(renderCited(result.answer || "", result.citations));
  const note = $(".review-note", box);
  const reviewed = result.review && result.review !== "not needed";
  note.hidden = !reviewed;
  note.textContent = reviewed ? `Review: ${result.review}.` : "";
  $(".agent-path", box).textContent = pathText(result);
  $(".agent-path", box).title = result.route_reason || "";
  box.hidden = false;
  $("#review").hidden = true;
}

function showReview(result) {
  state.pending = result;
  const box = $("#review");
  $(".question-echo", box).textContent = `“${state.question}”`;
  $(".answer-text", box).replaceChildren(renderCited(result.draft || "", result.citations));

  const rows = (result.flagged || []).map((f) => el("tr", {},
    el("td", {}, el("strong", {}, f.value), f.quote ? el("span", { class: "quote" }, `“${f.quote}”`) : null),
    el("td", {}, f.document || f.doc_id),
    el("td", {}, f.field),
    el("td", {}, el("span", { class: `evidence ${String(f.evidence).replace(/\s+/g, "-")}` }, f.evidence)),
    el("td", {}, f.doc_id && f.page
      ? el("button", { type: "button", class: "link-button", onclick: () => openViewer(f.doc_id, f.page, f.document) },
        `Page ${f.page}`)
      : "—"),
  ));
  $(".flagged tbody", box).replaceChildren(...rows);
  $(".flagged-wrap", box).hidden = !rows.length;
  const issues = (result.issues || []).map((i) => el("li", {}, i));
  $(".issues", box).replaceChildren(...issues);
  $(".issues-wrap", box).hidden = !issues.length;

  $(".edit-wrap", box).hidden = true;
  $('[data-action="edit"]', box).textContent = "Edit";
  $(".agent-path", box).textContent = pathText(result);
  $("#answer").hidden = true;
  box.hidden = false;
  $('[data-action="approve"]', box).focus({ preventScroll: true });
}

/* ---------------------------------------------------------------- page viewer */

function docTitle(docId, fallback) {
  const entry = state.chain.find((c) => c.doc_id === docId);
  return entry ? entry.title : fallback || docId;
}

function docPages(docId) {
  const entry = state.chain.find((c) => c.doc_id === docId);
  return entry && entry.num_pages ? entry.num_pages : null;
}

function openViewer(docId, page, title) {
  state.viewer = { docId, page: Number(page) || 1, pages: docPages(docId), title: title || docId };
  renderViewer();
  const dialog = $("#viewer");
  if (!dialog.open) dialog.showModal();
  $("#viewer-close").focus();
}

function renderViewer() {
  const v = state.viewer;
  const url = `/api/documents/${encodeURIComponent(v.docId)}/pages/${v.page}.png`;
  $("#viewer-title").textContent = v.title;
  $(".viewer-page").textContent = v.pages ? `Page ${v.page} of ${v.pages}` : `Page ${v.page}`;
  $('[data-step="-1"]').disabled = v.page <= 1;
  $('[data-step="1"]').disabled = v.pages ? v.page >= v.pages : false;
  $("#viewer-open").href = url;
  const img = $("#viewer-img");
  const status = $(".viewer-status");
  status.textContent = "Loading page…";
  img.alt = `${v.title}, page ${v.page}`;
  img.onload = () => { status.textContent = ""; };
  img.onerror = () => { status.textContent = "This page couldn't be loaded."; };
  img.src = url;
}

/* ---------------------------------------------------------------- wiring */

function init() {
  $("#examples").replaceChildren(...EXAMPLES.map((q) => el("li", {},
    el("button", { type: "button", class: "example", onclick: () => { $("#question").value = q; updateCount(); ask(q); } }, q))));

  const question = $("#question");
  const updateCount = () => { $("#question-count").textContent = `${question.value.length} / 500`; };
  question.addEventListener("input", updateCount);
  question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); ask(question.value); }
  });
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(question.value); });
  $("#parcel").addEventListener("change", (e) => selectParcel(e.target.value));
  document.querySelectorAll(".review-actions .button").forEach((b) => {
    b.addEventListener("click", () => submitReview(b.dataset.action));
  });

  const dialog = $("#viewer");
  $("#viewer-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); }); // backdrop click
  document.querySelectorAll(".viewer-nav [data-step]").forEach((b) => {
    b.addEventListener("click", () => {
      state.viewer.page += Number(b.dataset.step);
      renderViewer();
    });
  });

  updateControls();
  loadHealth();
  loadProperties();
}

document.addEventListener("DOMContentLoaded", init);
