/* Playground client: streams tokens from /api/generate over SSE and shows the
   chosen tier via /api/describe. Uses textContent only (no innerHTML) so model
   output is never interpreted as HTML. */

const messages = document.getElementById("messages");
const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
let busy = false;

async function loadDescribe() {
  try {
    const r = await fetch("/api/describe");
    if (!r.ok) throw new Error("no backend");
    const d = await r.json();
    const b = d.budget || {};
    set("d-tier", `${d.tier} (${d.tier_name})`);
    set("d-backing", d.backing);
    set("d-device", d.compute_device);
    set("d-cache", d.cache_layers ?? "-");
    set("d-weights", b.weights_gb != null ? `${b.weights_gb} GB` : "-");
    set("d-kv", b.kv_gb != null ? `${b.kv_gb} GB` : "-");
    const note = document.getElementById("d-note");
    note.textContent = d.honest_note || "";
    note.className = "verdict-note" + (d.tier === 3 ? " warn" : "");
  } catch (e) {
    showStaticPreview();
  }
}

// On a static host (GitHub Pages, no backend) the playground has no server to
// stream from. Explain how to run it locally instead of showing a dead UI.
function showStaticPreview() {
  set("d-tier", "not connected");
  const note = document.getElementById("d-note");
  note.textContent = "Static preview. The playground needs a running server.";
  note.className = "verdict-note warn";
  const empty = document.getElementById("empty");
  if (empty) {
    empty.textContent =
      "This is a static preview with no backend. Run the playground locally: " +
      "pip install 'streamllm[web]' then STREAMLLM_MODEL=<model-id> python web/server.py";
  }
  promptInput.placeholder = "Run the server locally to chat";
}

function set(id, v) { document.getElementById(id).textContent = v; }

function addMessage(role, text) {
  const empty = document.getElementById("empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

async function send() {
  if (busy) return;
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  busy = true;
  sendBtn.disabled = true;
  promptInput.value = "";
  addMessage("user", prompt);

  const bot = addMessage("bot", "");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bot.appendChild(cursor);

  try {
    const resp = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: 96, do_sample: true, temperature: 0.8 }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let acc = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop();
      for (const ev of events) handleEvent(ev, bot, cursor, (t) => { acc += t; });
    }
  } catch (e) {
    bot.textContent = `error: ${e.message}`;
  } finally {
    cursor.remove();
    busy = false;
    sendBtn.disabled = false;
    promptInput.focus();
  }
}

function handleEvent(raw, bot, cursor, onToken) {
  const lines = raw.split("\n");
  let kind = "", data = "";
  for (const l of lines) {
    if (l.startsWith("event: ")) kind = l.slice(7).trim();
    else if (l.startsWith("data: ")) data = l.slice(6);
  }
  if (!kind) return;
  let payload;
  try { payload = JSON.parse(data); } catch { payload = data; }

  if (kind === "token") {
    const node = document.createTextNode(payload);
    bot.insertBefore(node, cursor);
    messages.scrollTop = messages.scrollHeight;
  } else if (kind === "error") {
    bot.insertBefore(document.createTextNode(`\n[error: ${payload}]`), cursor);
  } else if (kind === "done") {
    const stats = document.getElementById("stats");
    stats.style.display = "flex";
    set("s-tps", `${payload.tokens_per_s} tok/s`);
    set("s-ttft", `TTFT ${payload.ttft_s}s`);
    set("s-bottleneck", payload.bottleneck);
  }
}

sendBtn.addEventListener("click", send);
promptInput.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
loadDescribe();
