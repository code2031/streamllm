/* Live tier calculator — a faithful client-side port of streamllm's memory
   budget model (src/streamllm/memory.py) and tiering policy (tiering.py).
   No backend: pick a model + hardware and see which tier streamllm would
   choose, plus the full peak-memory breakdown. The math here mirrors the
   Python exactly so the verdict matches `streamllm describe`. */

const GiB = 1024 ** 3;

// Real-ish configs for common model sizes (SwiGLU llama/qwen family).
const MODELS = {
  "1b":   { label: "1B",   hidden: 2048,  layers: 16,  heads: 32, kv: 8,  inter: 5632,  vocab: 128000, tied: true  },
  "7b":   { label: "7-8B", hidden: 4096,  layers: 32,  heads: 32, kv: 8,  inter: 14336, vocab: 128000, tied: false },
  "13b":  { label: "13B",  hidden: 5120,  layers: 40,  heads: 40, kv: 40, inter: 13824, vocab: 32000,  tied: false },
  "70b":  { label: "70B",  hidden: 8192,  layers: 80,  heads: 64, kv: 8,  inter: 28672, vocab: 128000, tied: false },
  "405b": { label: "405B", hidden: 16384, layers: 126, heads: 128, kv: 8, inter: 53248, vocab: 128000, tied: false },
};

// Available memory budgets for common machines (bytes computed from GB).
const HARDWARE = {
  "gpu8":    { label: "8 GB GPU + 16 GB RAM",    cuda: true,  vram: 8,   ram: 16,  unified: false },
  "gpu24":   { label: "24 GB GPU + 64 GB RAM",   cuda: true,  vram: 24,  ram: 64,  unified: false },
  "gpu80":   { label: "80 GB GPU + 256 GB RAM",  cuda: true,  vram: 80,  ram: 256, unified: false },
  "unified": { label: "128 GB unified (GB10 / Apple)", cuda: true, vram: 110, ram: 128, unified: true },
  "cpu":     { label: "CPU only, 32 GB RAM",     cuda: false, vram: 0,   ram: 32,  unified: false },
};

const DTYPE_BYTES = { bf16: 2, fp16: 2, fp32: 4 };
const QUANT_BPP = { none: null, int8: 1.0, int4: 0.5 };
const HEADROOM = 0.9;
const ACT_FACTOR = 2.5;
const MIN_CACHE = 2;

const state = {
  model: "7b", hardware: "gpu24", dtype: "bf16", quant: "none",
  context: 8192, batch: 1,
};

function estimate() {
  const m = MODELS[state.model];
  const headDim = Math.floor(m.hidden / m.heads);
  const dtypeBytes = DTYPE_BYTES[state.dtype];
  const wbpp = QUANT_BPP[state.quant] ?? dtypeBytes;

  const qDim = m.heads * headDim;
  const kvDim = m.kv * headDim;
  const attn = m.hidden * qDim + m.hidden * kvDim * 2 + qDim * m.hidden;
  const mlp = 3 * m.hidden * m.inter;
  const perLayerParams = attn + mlp + 2 * m.hidden;

  const embed = m.vocab * m.hidden;
  const lmHead = m.tied ? 0 : m.vocab * m.hidden;
  const residentParams = embed + m.hidden + lmHead;

  const perLayerBytes = perLayerParams * wbpp;
  const residentBytes = residentParams * wbpp;
  const weightsBytes = perLayerBytes * m.layers + residentBytes;

  const promptLen = Math.max(Math.min(Math.floor(state.context / 2), 2048), 1);
  const kvBytes = 2 * m.layers * m.kv * headDim * state.context * state.batch * dtypeBytes;
  const activationBytes = state.batch * promptLen * m.hidden * dtypeBytes * ACT_FACTOR;

  return { perLayerBytes, residentBytes, weightsBytes, kvBytes, activationBytes, layers: m.layers, kvHeads: m.kv };
}

function selectTier(est) {
  const hw = HARDWARE[state.hardware];
  const overhead = (hw.cuda ? 1.0 : 0.5) * GiB;
  const devAvail = hw.cuda ? (hw.unified ? Math.min(hw.vram, hw.ram) : hw.vram) * GiB : hw.ram * GiB;
  const ramAvail = hw.ram * GiB;
  const usableDev = devAvail * HEADROOM;
  const usableRam = ramAvail * HEADROOM;

  const tier0Peak = est.weightsBytes + est.kvBytes + est.activationBytes + overhead;
  const cacheDev = Math.floor(
    (usableDev - est.residentBytes - est.kvBytes - est.activationBytes - overhead) / est.perLayerBytes
  );
  const weightsFitRam = est.weightsBytes <= usableRam;

  let tier;
  if (tier0Peak <= usableDev) tier = 0;
  else if (weightsFitRam && hw.cuda && !hw.unified && cacheDev >= MIN_CACHE) tier = 1;
  else if (weightsFitRam) tier = 2;
  else tier = 3;

  return { tier, cacheDev, usableDev, usableRam, tier0Peak, weightsFitRam, overhead, hw };
}

const TIER_META = {
  0: { name: "full", cls: "t0",
       note: "Model fits in device memory. streamllm adds no value here. vLLM / TGI / llama.cpp will be faster. It loads normally and says so." },
  1: { name: "gpu_ram", cls: "t1",
       note: "Resident set on GPU, remaining decoder layers stream from pinned host RAM with an LRU cache and async prefetch." },
  2: { name: "ram", cls: "t2",
       note: "All decoder weights live in RAM and stream into compute. On unified memory this is residency management, not a real copy." },
  3: { name: "disk", cls: "t3", warn: true,
       note: "Disk-backed streaming is I/O-bound by physics: every layer is read per token. Quantized shards, batching and prefetch reduce the wall, none remove it." },
};

function gb(bytes) { return (bytes / 1e9).toFixed(2); }

function render() {
  const est = estimate();
  const d = selectTier(est);
  const meta = TIER_META[d.tier];

  document.getElementById("verdict-tier").textContent = `Tier ${d.tier}`;
  document.getElementById("verdict-tier").className = `verdict-tier ${meta.cls}`;
  const pill = document.getElementById("verdict-pill");
  pill.className = `tier-pill ${meta.cls}`;
  pill.innerHTML = `<span class="swatch"></span>${meta.name}`;

  const cl = (d.tier === 1 || d.tier === 2 || d.tier === 3)
    ? `, cache_layers=${Math.max(d.cacheDev, d.tier === 1 ? MIN_CACHE : 1)}` : "";
  document.getElementById("verdict-reason").textContent =
    `tier0_peak=${gb(d.tier0Peak)}GB ${d.tier0Peak <= d.usableDev ? "<=" : ">"} usable_device=${gb(d.usableDev)}GB; ` +
    `weights=${gb(est.weightsBytes)}GB ${d.weightsFitRam ? "<=" : ">"} usable_ram=${gb(d.usableRam)}GB${cl}`;

  const noteEl = document.getElementById("verdict-note");
  noteEl.textContent = meta.note;
  noteEl.className = "verdict-note" + (meta.warn ? " warn" : "");

  // memory breakdown bar (peak on device)
  const total = est.weightsBytes + est.kvBytes + est.activationBytes + d.overhead;
  const pct = (x) => `${Math.max((x / total) * 100, 0.5)}%`;
  document.getElementById("b-weights").style.width = pct(est.weightsBytes);
  document.getElementById("b-kv").style.width = pct(est.kvBytes);
  document.getElementById("b-act").style.width = pct(est.activationBytes);
  document.getElementById("b-over").style.width = pct(d.overhead);
  document.getElementById("bar-total").textContent = `${gb(total)} GB peak`;

  document.getElementById("bd-weights").textContent = `${gb(est.weightsBytes)} GB`;
  document.getElementById("bd-perlayer").textContent =
    `${est.layers} x ${(est.perLayerBytes / 1e9).toFixed(3)} GB`;
  document.getElementById("bd-resident").textContent = `${gb(est.residentBytes)} GB`;
  document.getElementById("bd-kv").textContent = `${gb(est.kvBytes)} GB  (kv_heads=${est.kvHeads})`;
  document.getElementById("bd-act").textContent = `${gb(est.activationBytes)} GB`;
  document.getElementById("bd-over").textContent = `${gb(d.overhead)} GB`;
}

function bindSeg(group, key) {
  document.querySelectorAll(`[data-seg="${group}"] button`).forEach((b) => {
    b.addEventListener("click", () => {
      state[key] = b.dataset.val;
      document.querySelectorAll(`[data-seg="${group}"] button`).forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      render();
    });
  });
}

function init() {
  const ms = document.getElementById("sel-model");
  for (const [k, v] of Object.entries(MODELS)) ms.add(new Option(v.label, k, false, k === state.model));
  const hs = document.getElementById("sel-hw");
  for (const [k, v] of Object.entries(HARDWARE)) hs.add(new Option(v.label, k, false, k === state.hardware));

  ms.addEventListener("change", () => { state.model = ms.value; render(); });
  hs.addEventListener("change", () => { state.hardware = hs.value; render(); });
  document.getElementById("inp-context").addEventListener("input", (e) => {
    state.context = Math.max(parseInt(e.target.value || "1", 10), 1); render();
  });
  document.getElementById("inp-batch").addEventListener("input", (e) => {
    state.batch = Math.max(parseInt(e.target.value || "1", 10), 1); render();
  });
  bindSeg("dtype", "dtype");
  bindSeg("quant", "quant");
  render();
}

document.addEventListener("DOMContentLoaded", init);
