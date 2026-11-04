/* SENTRY console — front end for app.py.
   Streaming updates arrive as Server-Sent Events; the canvas chart is hand
   drawn so it stays crisp at any size and needs no charting dependency. */

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const COLORS = {
    LOG: "#34d399",
    CHALLENGE: "#fbbf24",
    DROP: "#fb7185",
    UNKNOWN: "#94a3b8",
  };
  const LABELS = {
    LOG: "NO SYNTHETIC EVIDENCE",
    CHALLENGE: "SUSPICIOUS",
    DROP: "AI-GENERATED",
    UNKNOWN: "UNKNOWN",
  };
  const ACTIONS = {
    LOG: "Nothing in the validated feature set matched known synthesis. This is not proof of a human speaker.",
    CHALLENGE: "Ask the caller to repeat a random phrase before sharing anything.",
    DROP: "Do not trust this audio. Escalate or end the call.",
    UNKNOWN: "Outside the conditions this model was validated on. Do not treat this as authentic — verify another way.",
  };
  const IDLE_FOOT = "CPU-only · 1 s window";

  const stateOf = (decision) =>
    decision === "LOG"
      ? "log"
      : decision === "CHALLENGE"
      ? "challenge"
      : decision === "DROP"
      ? "drop"
      : decision === "UNKNOWN"
      ? "unknown"
      : "idle";

  const pct = (score) => Math.max(0, Math.min(100, Math.round(score * 100)));

  function fmtBytes(n) {
    if (!Number.isFinite(n)) return "";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function fmtDuration(seconds) {
    if (!Number.isFinite(seconds)) return "";
    if (seconds < 60) return `${seconds.toFixed(1)} s`;
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  }

  function toast(message, tone = "info", ms = 5200) {
    const stack = $("#toasts");
    const node = document.createElement("div");
    node.className = `toast${tone === "error" ? " is-error" : tone === "warning" ? " is-warning" : ""}`;
    node.textContent = message;
    stack.appendChild(node);
    window.setTimeout(() => {
      node.classList.add("is-leaving");
      window.setTimeout(() => node.remove(), 240);
    }, ms);
  }

  async function fetchJSON(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
  }

  /* ============================== verdict ============================== */

  class Verdict {
    constructor(root) {
      this.root = root;
      this.status = $('[data-role="status"]', root);
      this.meta = $('[data-role="meta"]', root);
      this.kicker = $('[data-role="kicker"]', root);
      this.label = $('[data-role="label"]', root);
      this.action = $('[data-role="action"]', root);
      this.score = $('[data-role="score"]', root);
      this.footLeft = $('[data-role="foot-left"]', root);
      this.footRight = $('[data-role="foot-right"]', root);
      this.fill = $('[data-role="fill"]', root);
      this.ladder = $$(".tier", root);
      this.progress = $(".gauge-progress", root);
      this.circumference = 2 * Math.PI * 51;
      this.progress.setAttribute("stroke-dasharray", this.circumference.toFixed(2));
      this.progress.setAttribute("stroke-dashoffset", this.circumference.toFixed(2));
      this.idleState = {
        status: this.status.textContent,
        meta: this.meta.textContent,
        kicker: this.kicker.textContent,
        label: this.label.textContent,
        action: this.action.textContent,
        footLeft: this.footLeft.textContent,
        footRight: this.footRight.textContent,
      };
    }

    setScore(score) {
      if (score == null || !Number.isFinite(score)) {
        this.score.textContent = "—";
        this.progress.setAttribute("stroke-dashoffset", this.circumference.toFixed(2));
        this.fill.style.width = "0%";
        return;
      }
      const value = pct(score);
      this.score.textContent = String(value);
      this.progress.setAttribute(
        "stroke-dashoffset",
        (this.circumference * (1 - value / 100)).toFixed(2)
      );
      this.fill.style.width = `${value}%`;
    }

    setTier(decision) {
      this.ladder.forEach((tier) => {
        tier.classList.toggle("is-on", tier.dataset.tier === decision);
      });
    }

    setText({ status, meta, kicker, label, action, footLeft, footRight }) {
      if (status != null) this.status.textContent = status;
      if (meta != null) this.meta.textContent = meta;
      if (kicker != null) this.kicker.textContent = kicker;
      if (label != null) this.label.textContent = label;
      if (action != null) this.action.textContent = action;
      if (footLeft != null) this.footLeft.textContent = footLeft;
      if (footRight != null) this.footRight.textContent = footRight;
    }

    reset(overrides = {}) {
      this.root.dataset.state = "idle";
      this.root.dataset.busy = "false";
      this.setText({ ...this.idleState, ...overrides });
      this.setScore(null);
      this.setTier(null);
    }

    busy({ label, action, meta, status = "Receiving", footRight = "0.0 s" }) {
      this.root.dataset.state = "idle";
      this.root.dataset.busy = "true";
      this.setText({
        status,
        meta: meta || "live",
        kicker: "SENTRY verdict",
        label: label || "Listening",
        action: action || "Speak naturally — the verdict updates four times per second.",
        footLeft: IDLE_FOOT,
        footRight,
      });
      this.setScore(null);
      this.setTier(null);
    }

    show({
      decision,
      score,
      status,
      meta,
      label,
      action,
      busy = false,
      footLeft = IDLE_FOOT,
      footRight,
    }) {
      const state = stateOf(decision);
      this.root.dataset.state = state;
      this.root.dataset.busy = busy ? "true" : "false";
      this.setText({
        status: status || decision,
        meta,
        kicker: "SENTRY verdict",
        label: label || LABELS[decision] || "Awaiting audio",
        action: action || ACTIONS[decision] || "",
        footLeft,
        footRight,
      });
      this.setScore(score);
      this.setTier(decision || null);
    }

    error(message) {
      this.root.dataset.state = "error";
      this.root.dataset.busy = "false";
      this.setText({
        status: "Error",
        meta: "no verdict",
        kicker: "SENTRY verdict",
        label: "Analysis failed",
        action: message,
        footLeft: IDLE_FOOT,
        footRight: "failed",
      });
      this.setScore(null);
      this.setTier(null);
    }
  }

  /* ============================== chart ================================ */

  class TraceChart {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.empty = $("[data-chart-empty]", canvas.parentElement);
      this.data = null;
      this.expected = null;
      this.hover = null;
      this.layout = null;
      this.w = 0;
      this.h = 0;

      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas.parentElement);
      canvas.addEventListener("pointermove", (event) => this.onPointer(event));
      canvas.addEventListener("pointerleave", () => {
        this.hover = null;
        this.draw();
      });
      this.resize();
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.w = Math.max(1, rect.width);
      this.h = Math.max(1, rect.height);
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.draw();
    }

    setData(data) {
      this.data = data;
      this.empty.hidden = Boolean(data && data.t && data.t.length);
      this.canvas.style.cursor = data && data.t && data.t.length ? "crosshair" : "default";
      this.draw();
    }

    clear() {
      this.data = null;
      this.hover = null;
      this.empty.hidden = false;
      this.canvas.style.cursor = "default";
      this.draw();
    }

    setExpectedDuration(seconds) {
      this.expected = seconds || null;
    }

    xMax() {
      if (this.expected) return this.expected;
      const t = this.data && this.data.t;
      const last = t && t.length ? t[t.length - 1] : 0;
      if (this.data && this.data.final) return Math.max(1, last + 0.15);
      return Math.max(1, last * 1.08 + 0.1);
    }

    onPointer(event) {
      if (!this.data || !this.layout || !this.data.t.length) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const { plot } = this.layout;
      if (x < plot.left || x > plot.right || y < plot.top || y > plot.bottom) {
        this.hover = null;
        this.draw();
        return;
      }
      const domainX = ((x - plot.left) / plot.width) * this.layout.xMax;
      let best = 0;
      let distance = Infinity;
      this.data.t.forEach((time, index) => {
        const delta = Math.abs(time - domainX);
        if (delta < distance) {
          distance = delta;
          best = index;
        }
      });
      this.hover = best;
      this.draw();
    }

    draw() {
      const ctx = this.ctx;
      const { w, h } = this;
      ctx.clearRect(0, 0, w, h);
      if (!w || !h) return;

      const compact = w < 520;
      const padLeft = compact ? 34 : 42;
      const padRight = 14;
      const padTop = 12;
      const padBottom = 26;
      const energyHeight = compact ? 34 : 42;
      const gap = 14;

      const plot = {
        left: padLeft,
        right: w - padRight,
        top: padTop + energyHeight + gap,
        bottom: h - padBottom,
      };
      plot.width = Math.max(20, plot.right - plot.left);
      plot.height = Math.max(20, plot.bottom - plot.top);

      const xMax = this.xMax();
      this.layout = { plot, xMax };
      const X = (t) =>
        plot.left + (Math.min(Math.max(t, 0), xMax) / xMax) * plot.width;
      const Y = (v) => plot.top + (1 - Math.max(0, Math.min(1, v))) * plot.height;

      /* probability bands */
      ctx.fillStyle = "rgba(52, 211, 153, 0.045)";
      ctx.fillRect(plot.left, Y(0.4), plot.width, Y(0) - Y(0.4));
      ctx.fillStyle = "rgba(251, 191, 36, 0.05)";
      ctx.fillRect(plot.left, Y(0.6), plot.width, Y(0.4) - Y(0.6));
      ctx.fillStyle = "rgba(251, 113, 133, 0.045)";
      ctx.fillRect(plot.left, Y(1), plot.width, Y(0.6) - Y(1));

      /* vertical + horizontal grid */
      ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
      ctx.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        ctx.beginPath();
        ctx.moveTo(plot.left, Math.round(Y(v)) + 0.5);
        ctx.lineTo(plot.right, Math.round(Y(v)) + 0.5);
        ctx.stroke();
      });

      const step =
        xMax <= 6 ? 1 : xMax <= 12 ? 2 : xMax <= 30 ? 5 : xMax <= 60 ? 10 : 20;
      ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = "#65758c";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        ctx.fillText(v.toFixed(2), plot.left - 8, Y(v));
      });

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let t = 0; t <= xMax + 0.001; t += step) {
        const x = Math.round(X(t)) + 0.5;
        ctx.strokeStyle = "rgba(148, 163, 184, 0.05)";
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.bottom);
        ctx.stroke();
        ctx.fillStyle = "#65758c";
        ctx.fillText(t === 0 ? "0" : `${t}s`, x, plot.bottom + 8);
      }

      /* thresholds */
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      [[0.4, "rgba(251, 191, 36, 0.5)"], [0.6, "rgba(251, 113, 133, 0.5)"]].forEach(
        ([v, color]) => {
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(plot.left, Y(v) + 0.5);
          ctx.lineTo(plot.right, Y(v) + 0.5);
          ctx.stroke();
        }
      );
      ctx.restore();

      if (!this.data || !this.data.t.length) return;

      const { t, p, rms, decisions } = this.data;

      /* energy strip */
      const energyTop = padTop;
      const energyBottom = energyTop + energyHeight;
      const peak = Math.max(...rms, 0.0001) * 1.18;
      ctx.beginPath();
      ctx.moveTo(X(t[0]), energyBottom);
      t.forEach((time, index) => {
        const value = Math.min(1, (rms[index] || 0) / peak);
        ctx.lineTo(X(time), energyBottom - value * energyHeight * 0.9);
      });
      ctx.lineTo(X(t[t.length - 1]), energyBottom);
      ctx.closePath();
      ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
      ctx.fill();

      ctx.beginPath();
      t.forEach((time, index) => {
        const value = Math.min(1, (rms[index] || 0) / peak);
        const x = X(time);
        const y = energyBottom - value * energyHeight * 0.9;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = "rgba(56, 189, 248, 0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();

      /* probability trace, coloured by emitted decision */
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      if (t.length === 1) {
        const color = COLORS[decisions[0]] || "#5eead4";
        ctx.beginPath();
        ctx.arc(X(t[0]), Y(p[0]), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      } else {
        for (let index = 1; index < t.length; index += 1) {
          const color = COLORS[decisions[index]] || "#5eead4";
          ctx.beginPath();
          ctx.moveTo(X(t[index - 1]), Y(p[index - 1]));
          ctx.lineTo(X(t[index]), Y(p[index]));
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.shadowColor = color;
          ctx.shadowBlur = 5;
          ctx.stroke();
        }
        ctx.shadowBlur = 0;
      }

      /* current point */
      const last = t.length - 1;
      const lastX = X(t[last]);
      const lastY = Y(p[last]);
      const lastColor = COLORS[decisions[last]] || "#5eead4";
      ctx.beginPath();
      ctx.arc(lastX, lastY, 7, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(6, 8, 12, 0.9)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = lastColor;
      ctx.shadowColor = lastColor;
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;

      /* hover readout */
      if (
        this.hover != null &&
        this.hover >= 0 &&
        this.hover < t.length &&
        this.data.t.length
      ) {
        const index = this.hover;
        const hx = X(t[index]);
        const hy = Y(p[index]);
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "rgba(233, 239, 248, 0.28)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(hx, plot.top);
        ctx.lineTo(hx, plot.bottom);
        ctx.stroke();
        ctx.restore();

        ctx.beginPath();
        ctx.arc(hx, hy, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#f8fafc";
        ctx.fill();

        const decision = decisions[index] || "LOG";
        const lines = [
          `${t[index].toFixed(1)} s`,
          `spoof ${pct(p[index])}%`,
          decision,
        ];
        const boxWidth = 116;
        const boxHeight = 58;
        let boxX = hx + 12;
        if (boxX + boxWidth > plot.right) boxX = hx - boxWidth - 12;
        let boxY = hy - boxHeight / 2;
        boxY = Math.max(plot.top + 4, Math.min(plot.bottom - boxHeight - 4, boxY));

        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 8);
        else ctx.rect(boxX, boxY, boxWidth, boxHeight);
        ctx.fillStyle = "rgba(10, 16, 26, 0.96)";
        ctx.fill();
        ctx.strokeStyle = "rgba(148, 163, 184, 0.24)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "#dbe6f3";
        ctx.fillText(lines[0], boxX + 10, boxY + 9);
        ctx.fillStyle = "#96a6bd";
        ctx.fillText(lines[1], boxX + 10, boxY + 24);
        ctx.fillStyle = COLORS[decision] || "#5eead4";
        ctx.font = "700 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(lines[2], boxX + 10, boxY + 39);
      }
    }
  }

  /* ========================= analysis controller ======================== */

  function createAnalysisController(refs) {
    let source = null;
    let onDone = null;
    let active = false;

    function finish() {
      active = false;
      const callback = onDone;
      onDone = null;
      if (callback) callback();
    }

    function stop() {
      if (source) {
        source.close();
        source = null;
      }
    }

    function removeWarning() {
      const old = refs.evidence.querySelector(".warn-box");
      if (old) old.remove();
    }

    function renderEvidence(payload) {
      const s = payload.summary;
      const state = stateOf(s.decision);
      const items = [
        ["Speech windows", String(s.n), ""],
        ["Window mean", s.mean.toFixed(2), ""],
        ["Top-30% score", s.top.toFixed(2), `state-${state}`],
        ["Top-10% score", s.top10.toFixed(2), s.sparse ? "warn" : ""],
        ["AI windows", `${Math.round(s.frac_drop * 100)}%`, ""],
      ];
      if (Number.isFinite(s.ood_frac)) {
        items.push([
          "In-scope windows",
          `${Math.round((1 - s.ood_frac) * 100)}%`,
          s.out_of_scope ? "warn" : "",
        ]);
      }

      refs.metrics.replaceChildren(
        ...items.map(([label, value, className]) => {
          const node = document.createElement("div");
          node.className = `ev${className ? ` ${className}` : ""}`;
          const labelNode = document.createElement("span");
          labelNode.textContent = label;
          const valueNode = document.createElement("b");
          valueNode.textContent = value;
          node.append(labelNode, valueNode);
          return node;
        })
      );

      removeWarning();
      if (s.out_of_scope) {
        const warning = document.createElement("div");
        warning.className = "warn-box";
        warning.innerHTML =
          '<svg class="icon"><use href="#i-alert"></use></svg>' +
          "<span><b>Outside validated conditions.</b> Only " +
          `<b>${Math.round((1 - s.ood_frac) * 100)}%</b> of windows resemble the ` +
          "training data, so the model's score is not trustworthy here. This is " +
          "<b>unresolved — do not treat it as authentic</b>. Verify by another channel.</span>";
        refs.evidence.insertBefore(warning, refs.evidence.querySelector(".evidence-foot"));
      }
      if (s.sparse) {
        const warning = document.createElement("div");
        warning.className = "warn-box";
        warning.innerHTML =
          '<svg class="icon"><use href="#i-alert"></use></svg>' +
          "<span><b>Sparse synthetic evidence.</b> A few windows score strongly AI " +
          `(top-10% <b>${s.top10.toFixed(2)}</b>), but only <b>${Math.round(
            s.frac_drop * 100
          )}%</b> pass 0.60. Treat this clip as <b>unresolved — do not treat it as ` +
          "authentic</b>.</span>";
        refs.evidence.insertBefore(warning, refs.evidence.querySelector(".evidence-foot"));
      }

      refs.sub.textContent = payload.title || "Aggregation is over speech-bearing windows only.";
      refs.rule.textContent = s.out_of_scope
        ? "outside validated set · novelty guard"
        : s.sparse
        ? "sparse-evidence guard · top-10% ≥ 0.45"
        : "top-30% + AI-window ratio aggregation";
      refs.decision.dataset.state = state;
      refs.decision.textContent = s.decision;
      refs.report.hidden = !payload.report_url;
      if (payload.report_url) {
        refs.report.href = payload.report_url;
        refs.report.setAttribute("download", payload.report_name || "report.json");
      }
      refs.evidence.hidden = false;
    }

    function start(fid, options = {}) {
      stop();
      active = true;
      onDone = options.onDone || null;
      if (options.onStart) options.onStart();

      refs.chart.clear();
      refs.chart.setExpectedDuration(null);
      refs.evidence.hidden = true;
      refs.report.hidden = true;
      removeWarning();
      refs.verdict.busy({
        status: "Analysing",
        label: "Reading the clip",
        action: options.action || "Scoring one-second windows and aggregating the evidence.",
        meta: options.meta || "streaming",
        footRight: "0.0 s",
      });

      let finished = false;
      source = new EventSource(`/api/analyze/${encodeURIComponent(fid)}`);

      source.onmessage = (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }
        if (payload.error) {
          finished = true;
          stop();
          refs.verdict.error(payload.error);
          toast(payload.error, "error");
          finish();
          return;
        }

        refs.chart.setData(payload);
        const final = Boolean(payload.final);
        const decision =
          final && payload.summary ? payload.summary.decision : payload.decision;
        const score =
          final && payload.summary ? payload.summary.score : payload.smooth ?? payload.score;
        const latency = Number.isFinite(payload.latency_ms)
          ? `${payload.latency_ms.toFixed(1)} ms / window`
          : "—";

        refs.verdict.show({
          decision,
          score,
          status: final ? decision : `${decision} · live`,
          meta: final
            ? payload.title || options.meta || "clip"
            : `${payload.now.toFixed(1)} s · ${options.meta || "streaming"}`,
          busy: !final,
          footRight: latency,
        });

        if (final) {
          finished = true;
          stop();
          if (payload.summary) renderEvidence(payload);
          finish();
        }
      };

      source.onerror = () => {
        if (finished) return;
        finished = true;
        stop();
        const message =
          "The analysis stream stopped unexpectedly. Check the server log and try again.";
        refs.verdict.error(message);
        toast(message, "error");
        finish();
      };
    }

    function reset() {
      const wasActive = active;
      stop();
      if (wasActive) finish();
      active = false;
      onDone = null;
      removeWarning();
      refs.chart.clear();
      refs.chart.setExpectedDuration(null);
      refs.evidence.hidden = true;
      refs.report.hidden = true;
      refs.verdict.reset();
    }

    return { start, reset, stop, isActive: () => active };
  }

  /* ============================== elements ============================= */

  const tabs = $$(".tab");
  const views = {
    live: $("#view-live"),
    file: $("#view-file"),
    attack: $("#view-attack"),
    method: $("#view-method"),
  };

  const liveVerdict = new Verdict($("#live-verdict"));
  const fileVerdict = new Verdict($("#file-verdict"));
  const attackVerdict = new Verdict($("#attack-verdict"));

  const liveChart = new TraceChart($("#live-chart"));
  const fileChart = new TraceChart($("#file-chart"));
  const attackChart = new TraceChart($("#attack-chart"));

  const fileController = createAnalysisController({
    verdict: fileVerdict,
    chart: fileChart,
    evidence: $("#file-evidence"),
    metrics: $("#file-metrics"),
    decision: $("#file-decision"),
    rule: $("#file-evidence-rule"),
    report: $("#file-report"),
    sub: $("#file-evidence-sub"),
  });

  const attackController = createAnalysisController({
    verdict: attackVerdict,
    chart: attackChart,
    evidence: $("#attack-evidence"),
    metrics: $("#attack-metrics"),
    decision: $("#attack-decision"),
    rule: $("#attack-evidence-rule"),
    report: $("#attack-report"),
    sub: $("#attack-evidence-sub"),
  });

  /* ============================== tabs ================================= */

  function activateTab(name, updateHash = true) {
    if (!views[name]) return;
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    Object.entries(views).forEach(([key, view]) => {
      const active = key === name;
      view.hidden = !active;
      view.classList.toggle("is-active", active);
    });
    if (updateHash && window.location.hash !== `#${name}`) {
      window.history.replaceState(null, "", `#${name}`);
    }
    window.requestAnimationFrame(() => {
      if (name === "live") liveChart.resize();
      if (name === "file") fileChart.resize();
      if (name === "attack") attackChart.resize();
    });
  }

  tabs.forEach((tab) => {
    const name = tab.dataset.tab;
    tab.id = `tab-${name}`;
    tab.tabIndex = tab.classList.contains("is-active") ? 0 : -1;
    tab.addEventListener("click", () => activateTab(name));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const index = tabs.indexOf(tab);
      let next = index;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      tabs[next].focus();
      activateTab(tabs[next].dataset.tab);
    });
  });

  const initial = window.location.hash.replace("#", "");
  if (views[initial]) activateTab(initial, false);

  window.addEventListener("hashchange", () => {
    const name = window.location.hash.replace("#", "");
    if (views[name]) activateTab(name, false);
  });

  /* ============================== header =============================== */

  function setMetric(name, text) {
    $$(`[data-metric="${name}"]`).forEach((node) => {
      node.textContent = text;
    });
  }

  async function loadHealth() {
    try {
      const data = await fetchJSON("/api/health");
      const r = data.results || {};
      const pct1 = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
      const pct2 = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : "—");

      setMetric("accuracy", pct1(r.accuracy));
      setMetric("auc", Number.isFinite(r.auc) ? r.auc.toFixed(4) : "—");
      setMetric("eer", pct2(r.eer));
      setMetric(
        "holdout_n",
        Number.isFinite(r.holdout_n) ? r.holdout_n.toLocaleString() : "—"
      );
      setMetric("unseen", pct1(r.unseen_tts_accuracy));
      setMetric("channel", pct1(r.channel_accuracy));
      if (Number.isFinite(r.real_drop) && Number.isFinite(r.real_n)) {
        setMetric("real_fp", `${r.real_drop} / ${r.real_n}`);
      }
      if (Number.isFinite(r.tts_drop) && Number.isFinite(r.tts_n)) {
        setMetric("tts_drop", `${r.tts_drop} / ${r.tts_n}`);
      }
      if (Number.isFinite(r.lovo_min) && Number.isFinite(r.lovo_max)) {
        setMetric(
          "lovo",
          `${(r.lovo_min * 100).toFixed(1)}–${(r.lovo_max * 100).toFixed(0)}%`
        );
      }

      const accuracy = r.accuracy;
      $("#engine-meta").textContent = `${data.n_features} features · ${
        Number.isFinite(accuracy)
          ? `${(accuracy * 100).toFixed(1)}% held-out`
          : "CPU"
      }`;
    } catch {
      $("#engine-meta").textContent = "model unavailable";
    }
  }

  /* ============================== microphone =========================== */

  const micDevice = $("#mic-device");
  const micButton = $("#mic-start");
  const micState = $("#live-state");
  const durationButtons = $$(".seg");
  let micSource = null;
  let micActive = false;
  let micLast = null;
  let micDuration = 20;

  async function loadDevices() {
    try {
      const data = await fetchJSON("/api/devices");
      micDevice.replaceChildren(
        ...data.devices.map((device) => {
          const option = document.createElement("option");
          option.value = device.value;
          option.textContent = device.label;
          return option;
        })
      );
      if (data.default) micDevice.value = data.default;
    } catch {
      micDevice.replaceChildren(new Option("System default", "default"));
    }
  }

  durationButtons.forEach((button) => {
    button.addEventListener("click", () => {
      durationButtons.forEach((item) => item.classList.toggle("is-active", item === button));
      micDuration = Number(button.dataset.duration) || 20;
      if (!micActive) liveChart.setExpectedDuration(micDuration);
    });
  });

  function setMicButton(active) {
    micButton.classList.toggle("is-running", active);
    $(".btn-text", micButton).textContent = active ? "Stop listening" : "Start listening";
    const use = $(".btn-icon use", micButton);
    if (use) use.setAttribute("href", active ? "#i-stop" : "#i-mic");
  }

  function stopMic(reason = "stopped") {
    if (micSource) {
      micSource.close();
      micSource = null;
    }
    if (!micActive) return;
    micActive = false;
    setMicButton(false);
    if (reason === "complete") {
      micState.textContent = "complete";
    } else if (reason === "error") {
      micState.textContent = "failed";
    } else {
      micState.textContent = "stopped";
      if (micLast) {
        liveVerdict.show({
          decision: micLast.decision,
          score: micLast.smooth ?? micLast.score,
          status: "STOPPED",
          meta: `${micLast.now.toFixed(1)} s captured`,
          busy: false,
          footRight: Number.isFinite(micLast.latency_ms)
            ? `${micLast.latency_ms.toFixed(1)} ms / window`
            : "—",
        });
      }
    }
  }

  function startMic() {
    if (micActive) {
      stopMic("stopped");
      return;
    }
    const device = micDevice.value || "default";
    const label =
      micDevice.selectedOptions && micDevice.selectedOptions[0]
        ? micDevice.selectedOptions[0].textContent
        : device;
    micActive = true;
    micLast = null;
    setMicButton(true);
    micState.textContent = "capturing";
    liveChart.clear();
    liveChart.setExpectedDuration(micDuration);
    liveVerdict.busy({
      status: "Listening",
      label: "Capturing audio",
      action: "Speak naturally. The verdict updates four times per second.",
      meta: label,
      footRight: "0.0 s",
    });

    let finished = false;
    micSource = new EventSource(
      `/api/mic?duration=${encodeURIComponent(micDuration)}&device=${encodeURIComponent(device)}`
    );

    micSource.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }
      if (payload.error) {
        finished = true;
        stopMic("error");
        liveVerdict.error(payload.error);
        toast(payload.error, "error");
        return;
      }

      micLast = payload;
      liveChart.setData(payload);
      const final = Boolean(payload.final);
      const latency = Number.isFinite(payload.latency_ms)
        ? `${payload.latency_ms.toFixed(1)} ms / window`
        : "—";

      liveVerdict.show({
        decision: payload.decision,
        score: payload.smooth ?? payload.score,
        status: final ? payload.decision : `LIVE · ${payload.decision}`,
        meta: `${payload.now.toFixed(1)} s / ${micDuration} s`,
        busy: !final,
        footRight: latency,
      });
      micState.textContent = final
        ? "complete"
        : `${payload.now.toFixed(1)} s`;

      if (final) {
        finished = true;
        stopMic("complete");
      }
    };

    micSource.onerror = () => {
      if (finished) return;
      finished = true;
      stopMic("error");
      const message =
        "The microphone stream stopped. Check the input device and try again.";
      liveVerdict.error(message);
      toast(message, "error");
    };
  }

  micButton.addEventListener("click", startMic);

  /* ============================== file tab ============================= */

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const fileInfo = $("#file-info");
  const fileName = $("#file-name");
  const fileSize = $("#file-size");
  const fileAudio = $("#file-audio");
  const fileAnalyze = $("#file-analyze");
  const fileClear = $("#file-clear");
  const sampleList = $("#sample-list");
  let currentFile = null;

  function setSourceLoading(loading) {
    fileAnalyze.classList.toggle("is-loading", loading);
    fileAnalyze.disabled = loading || !currentFile;
  }

  function updateFileMeta() {
    if (!currentFile) return;
    const bits = [];
    if (currentFile.size) bits.push(fmtBytes(currentFile.size));
    else bits.push("held-out sample");
    if (Number.isFinite(currentFile.duration)) bits.push(fmtDuration(currentFile.duration));
    fileSize.textContent = bits.join(" · ");
  }

  function markSample(id) {
    $$(".sample", sampleList).forEach((button) => {
      button.classList.toggle("is-selected", Boolean(id) && button.dataset.sampleId === id);
    });
  }

  function loadSource({ id, name, size, audioUrl }) {
    currentFile = { id, name, size, audioUrl, duration: NaN };
    markSample(id && id.startsWith("sample-") ? id : null);
    fileController.reset();
    fileInfo.hidden = false;
    dropzone.hidden = true;
    fileName.textContent = name;
    fileName.title = name;
    updateFileMeta();
    fileAudio.src = audioUrl;
    fileAudio.load();
    fileAnalyze.disabled = false;
    fileVerdict.reset({
      status: "File loaded",
      meta: name,
      label: "Ready to analyze",
      action: "Press Analyze file to score this recording window by window.",
      footRight: "ready",
    });
  }

  function clearSource() {
    currentFile = null;
    markSample(null);
    fileController.reset();
    fileInfo.hidden = true;
    dropzone.hidden = false;
    fileAudio.removeAttribute("src");
    fileAudio.load();
    fileName.textContent = "clip.wav";
    fileSize.textContent = "—";
    fileAnalyze.disabled = true;
    fileInput.value = "";
  }

  fileAudio.addEventListener("loadedmetadata", () => {
    if (currentFile && Number.isFinite(fileAudio.duration)) {
      currentFile.duration = fileAudio.duration;
      updateFileMeta();
    }
  });

  fileClear.addEventListener("click", clearSource);

  async function uploadFile(file) {
    if (!file) return;
    setSourceLoading(true);
    fileAnalyze.classList.add("is-loading");
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await fetchJSON("/api/upload", { method: "POST", body: form });
      loadSource({
        id: data.id,
        name: data.filename,
        size: data.size,
        audioUrl: data.audio_url,
      });
      toast("File ready. Press Analyze when you are ready.", "info", 3800);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      fileInput.value = "";
      fileAnalyze.classList.remove("is-loading");
      fileAnalyze.disabled = !currentFile;
    }
  }

  fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-drag");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-drag");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  fileAnalyze.addEventListener("click", () => {
    if (!currentFile) return;
    fileController.start(currentFile.id, {
      meta: currentFile.name,
      action: "Scoring one-second windows and aggregating the clip evidence.",
      onStart: () => setSourceLoading(true),
      onDone: () => setSourceLoading(false),
    });
  });

  async function loadSamples() {
    try {
      const data = await fetchJSON("/api/samples");
      if (!data.samples.length) {
        sampleList.textContent = "No local samples found in data/.";
        return;
      }
      sampleList.replaceChildren(
        ...data.samples.map((sample) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "sample";
          button.dataset.sampleId = sample.id;
          button.dataset.tone = sample.id.replace("sample-", "");
          button.innerHTML = '<i class="dot"></i>';
          const copy = document.createElement("span");
          const label = document.createElement("b");
          label.textContent = sample.label;
          const description = document.createElement("span");
          description.textContent = `${sample.description} · ${sample.filename}`;
          copy.append(label, description);
          button.append(copy);
          button.addEventListener("click", () => {
            loadSource({
              id: sample.id,
              name: sample.filename,
              size: 0,
              audioUrl: sample.audio_url,
            });
            fileController.start(sample.id, {
              meta: sample.label,
              onStart: () => setSourceLoading(true),
              onDone: () => setSourceLoading(false),
            });
          });
          return button;
        })
      );
    } catch {
      sampleList.textContent = "Samples unavailable.";
    }
  }

  /* ============================== attack lab =========================== */

  const ttsText = $("#tts-text");
  const ttsVoice = $("#tts-voice");
  const ttsButton = $("#tts-generate");
  const ttsGenerated = $("#tts-generated");
  const ttsAudio = $("#tts-audio");
  const ttsGeneratedName = $("#tts-generated-name");

  $$(".preset").forEach((button) => {
    button.addEventListener("click", () => {
      ttsText.value = button.dataset.text || ttsText.value;
      ttsVoice.value = button.dataset.voice || ttsVoice.value;
      ttsText.focus();
    });
  });

  function setTtsLoading(loading) {
    ttsButton.classList.toggle("is-loading", loading);
    ttsButton.disabled = loading;
  }

  ttsButton.addEventListener("click", async () => {
    const text = ttsText.value.trim();
    const voice = ttsVoice.value;
    if (!text) {
      toast("Enter a script first.", "warning");
      ttsText.focus();
      return;
    }

    attackController.reset();
    setTtsLoading(true);
    attackVerdict.busy({
      status: "Synthesising",
      label: "Generating voice",
      action: "edge-tts is rendering the script, then SENTRY will score it.",
      meta: voice,
      footRight: "network",
    });

    try {
      const data = await fetchJSON("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice }),
      });
      ttsGenerated.hidden = false;
      ttsGeneratedName.textContent = data.filename;
      ttsAudio.src = data.audio_url;
      ttsAudio.load();
      attackController.start(data.id, {
        meta: voice,
        action: "Scoring the generated clip window by window.",
        onDone: () => setTtsLoading(false),
      });
    } catch (error) {
      attackVerdict.error(error.message);
      toast(error.message, "error", 8000);
      setTtsLoading(false);
    }
  });

  /* ============================== boot ================================= */

  loadHealth();
  loadDevices();
  loadSamples();
})();
