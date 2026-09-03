/* ==========================================================================
   calcia showcase -- the whole point of this file is what it does NOT do.
   No framework, no WebGL, no per-frame geometry. It swaps images, moves a clip
   rectangle, draws a few hundred circles on demand, and lets the browser's
   video decoder do the only heavy lifting on the page.
   ========================================================================== */
(function (global) {
  "use strict";

  var A = "assets/";
  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Inlined by web/bundle_data.py so the page runs off the filesystem with no
  // server; when it is absent we simply fetch the same files.
  var INLINE = global.CALCIA_DATA || null;

  function get(url, key) {
    if (INLINE && key && INLINE[key]) return Promise.resolve(INLINE[key]);
    return fetch(url, { cache: "force-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return r.json();
    });
  }
  function $(sel, root) { return (root || document).querySelector(sel); }
  function el(tag, cls) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    return n;
  }
  function fmt(n) { return n.toLocaleString("en-US"); }

  /* ---------------------------------------------------------------- language */
  function initLang() {
    var stored = null;
    try { stored = localStorage.getItem("calcia-lang"); } catch (e) {}
    var initial = stored ||
      ((navigator.language || "en").toLowerCase().indexOf("zh") === 0 ? "zh" : "en");
    setLang(initial);
    Array.prototype.forEach.call(
      document.querySelectorAll("[data-set-lang]"), function (btn) {
        btn.addEventListener("click", function () {
          setLang(btn.getAttribute("data-set-lang"));
        });
      });
  }
  function setLang(lang) {
    document.body.setAttribute("data-lang", lang);
    document.documentElement.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
    Array.prototype.forEach.call(
      document.querySelectorAll("[data-set-lang]"), function (b) {
        b.classList.toggle("is-on", b.getAttribute("data-set-lang") === lang);
      });
    try { localStorage.setItem("calcia-lang", lang); } catch (e) {}
  }

  /* ------------------------------------------------- viewport video gating
     A page with six autoplaying videos will heat a laptop even though each
     decode is cheap. Only what is actually on screen is allowed to run, and
     nothing buffers until it is close to being seen. */
  var gated = [];
  function gateVideo(video) {
    gated.push(video);
    if (!("IntersectionObserver" in window)) { tryPlay(video); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          if (video.preload === "none") video.preload = "auto";
          tryPlay(video);
        } else {
          video.pause();
        }
      });
    }, { rootMargin: "160px 0px", threshold: 0.12 });
    io.observe(video);
  }
  function tryPlay(v) {
    if (reduceMotion) return;
    var p = v.play();
    if (p && p.catch) p.catch(function () { /* autoplay blocked; poster stays */ });
  }
  function setSources(video, webm, mp4, poster) {
    if (poster) video.poster = A + poster;
    [["video/webm", webm], ["video/mp4", mp4]].forEach(function (pair) {
      if (!pair[1]) return;
      var s = el("source");
      s.type = pair[0];
      s.src = A + pair[1];
      video.appendChild(s);
    });
    video.load();
  }

  /* -------------------------------------------------------------------- hero */
  var BEAT_TEXT = {
    dust_in: {
      k: { en: "the growth field", zh: "生长场" },
      t: {
        en: "Attractor points scattered through the tissue. Dendrites are not routed — they compete for these.",
        zh: "散布在组织里的吸引子。树突不是被布线的——它们要去争夺这些点。"
      }
    },
    bloom: {
      k: { en: "somata", zh: "胞体" },
      t: {
        en: "Cell bodies appear where the exclusion sampler put them — never closer to each other than the minimum separation, never inside a vessel.",
        zh: "胞体出现在排斥采样器安放它们的位置——彼此不会近于最小间距，也不会落在血管里。"
      }
    },
    grow: {
      k: { en: "space colonization", zh: "空间占领" },
      t: {
        en: "Every tree grows toward nearby attractors at once, from one shared pool. Whoever reaches a point first consumes it — that competition alone is what keeps the trees out of each other.",
        zh: "所有树同时朝附近的吸引子生长，共用同一个池子。谁先到就消耗掉它——仅凭这种竞争，树与树就自然避开了彼此。"
      }
    },
    axons_in: {
      k: { en: "neuropil", zh: "神经毡" },
      t: {
        en: "Background processes drift in. In one-photon imaging this is what washes the picture out.",
        zh: "背景突起浮现。在单光子成像里，正是它把画面冲淡。"
      }
    },
    pulse: {
      k: { en: "calcium", zh: "钙" },
      t: {
        en: "Poisson bursts, driven through the indicator's own impulse response.",
        zh: "泊松放电簇，经由指示剂自身的冲激响应驱动。"
      }
    }
  };
  var BEAT_ORDER = ["dust_in", "bloom", "grow", "axons_in", "pulse"];

  function initHero(growth) {
    var canvas = $("#growth-canvas");
    var cap = $("#hero-caption");
    var track = $("#hero-track");
    if (!canvas || !growth) return;

    // stats strip
    var stats = $("#hero-stats");
    var s = growth.scene || {};
    [[fmt(s.n_dendrite_nodes || 0), { en: "dendrite nodes", zh: "树突节点" }],
     [fmt(s.n_growth_iterations || 0), { en: "growth steps", zh: "生长步数" }],
     [fmt(s.n_attractors || 0), { en: "attractors consumed", zh: "被消耗的吸引子" }],
     [(s.volume_um || []).join(" × ") + " µm", { en: "tissue block", zh: "组织块" }],
     [String(s.n_neurons || 0), { en: "neurons on screen", zh: "画面中的神经元" }]
    ].forEach(function (row) {
      var d = el("div"), b = el("b"), sp = el("span");
      b.textContent = row[0];
      sp.innerHTML = '<span class="en">' + row[1].en + '</span>' +
                     '<span class="zh">' + row[1].zh + '</span>';
      d.appendChild(b); d.appendChild(sp); stats.appendChild(d);
    });

    var beats = BEAT_ORDER.filter(function (k) { return growth.beats[k]; })
      .map(function (k) {
        var b = growth.beats[k];
        var seg = el("i");
        seg.style.setProperty("--f", (b.end - b.start).toFixed(4));
        track.appendChild(seg);
        return { key: k, start: b.start, end: b.end, seg: seg };
      });

    var current = null, lastScrub = -1;
    function onTime(t) {
      var active = null;
      beats.forEach(function (b) {
        var p = Math.max(0, Math.min(1, (t - b.start) / Math.max(b.end - b.start, 1e-6)));
        var v = p.toFixed(3);
        if (b.last !== v) { b.seg.style.setProperty("--p", v); b.last = v; }
        if (t >= b.start && t < b.end) active = b;
      });
      if (active && active.key !== current) {
        current = active.key;
        var txt = BEAT_TEXT[current];
        cap.innerHTML =
          '<b><span class="en">' + txt.k.en + '</span><span class="zh">' + txt.k.zh + '</span></b>' +
          '<span class="en">' + txt.t.en + '</span>' +
          '<span class="zh">' + txt.t.zh + '</span>';
        cap.classList.add("is-on");
      } else if (!active && current) {
        current = null;
        cap.classList.remove("is-on");
      }
      var sv = Math.round(t * 1000);
      if (!scrubbing && sv !== lastScrub) { scrubEl.value = String(sv); lastScrub = sv; }
    }

    var scrubEl = $("#hero-scrub"), playEl = $("#hero-play"), scrubbing = false;

    loadGrowthScene(canvas, A + "growth/", { bloom: true }, INLINE).then(function (r) {
      var sc = r.scene;
      sc.playing = !reduceMotion;
      sc.attach({ onTime: onTime });

      // Reduced motion: hold a still of the grown scene instead of looping.
      // This is a DEFAULT, so it is applied before ?t= gets a chance to override.
      if (reduceMotion) {
        playEl.firstElementChild.className = "ico-play";
        sc.scrub = 0.75;
        sc.autoSpin = false;
      }

      // ?t=0.62 freezes the scene at a point in the loop -- handy for grabbing
      // a still, and it makes a moment in the growth linkable.
      var q = /[?&]t=([0-9.]+)/.exec(location.search);
      if (q) {
        sc.scrub = Math.max(0, Math.min(1, parseFloat(q[1])));
        sc.playing = false;
        playEl.firstElementChild.className = "ico-play";
      }
      if (sc.scrub !== null) {
        sc.render(sc.scrub);
        onTime(sc.scrub);
      }
      sc.start();

      playEl.addEventListener("click", function () {
        sc.playing = !sc.playing;
        if (sc.playing) {
          sc.scrub = null; scrubbing = false;
          // Reduced motion parks the camera; asking for play is asking for all
          // of it back, orbit included.
          sc.autoSpin = true; sc.idle = 99;
        }
        playEl.firstElementChild.className = sc.playing ? "ico-pause" : "ico-play";
        playEl.setAttribute("aria-label", sc.playing ? "pause" : "play");
        sc.start();
      });
      // Scrubbing is the thing a video cannot do: the growth is recomputed
      // from the same buffer at whatever iteration you point at.
      function onScrub() {
        scrubbing = true;
        sc.playing = false;
        sc.scrub = scrubEl.value / 1000;
        sc.time = sc.scrub;
        playEl.firstElementChild.className = "ico-play";
        sc.start();
      }
      scrubEl.addEventListener("input", onScrub);
      scrubEl.addEventListener("change", function () { scrubbing = false; });
    }).catch(function (err) {
      // No WebGL, or the payload is missing: fall back to the rendered still.
      console.warn("live growth scene unavailable:", err);
      canvas.hidden = true;
      var img = $("#hero-fallback");
      if (img && growth.poster) {
        img.src = A + "growth/" + growth.poster;
        img.hidden = false;
      }
      $("#hero-ctrl").hidden = true;
    });
  }

  /* ----------------------------------------------------------------- compare
     Two videos, one clip rectangle. Kept in sync only when they actually
     drift, because forcing currentTime every frame causes stutter. */
  function initCompare(manifest) {
    var wrap = $("#compare");
    if (!wrap) return;
    var chans = manifest.channels || {};
    var left = chans.gcamp_crisp, right = chans.gcamp_noisy;
    if (!left || !right) { wrap.remove(); return; }

    var a = $(".cmp-a", wrap), b = $(".cmp-b", wrap), handle = $("#cmp-handle");
    setSources(a, left.webm, left.mp4, left.poster);
    setSources(b, right.webm, right.mp4, right.poster);
    gateVideo(a); gateVideo(b);

    a.addEventListener("timeupdate", function () {
      if (Math.abs(a.currentTime - b.currentTime) > 0.18) b.currentTime = a.currentTime;
    });

    function setSplit(clientX) {
      var r = wrap.getBoundingClientRect();
      var p = Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100));
      wrap.style.setProperty("--split", p + "%");
    }
    wrap.style.setProperty("--split", "50%");
    var down = false;
    handle.addEventListener("pointerdown", function (e) {
      down = true; handle.setPointerCapture(e.pointerId);
    });
    handle.addEventListener("pointermove", function (e) { if (down) setSplit(e.clientX); });
    handle.addEventListener("pointerup", function () { down = false; });
    wrap.addEventListener("pointerdown", function (e) {
      if (e.target !== handle && !handle.contains(e.target)) setSplit(e.clientX);
    });
  }

  /* ------------------------------------------------------------ ground truth
     A few hundred circles on a canvas, redrawn only when the threshold moves
     or the element is resized. Nothing runs per video frame. */
  function initGroundTruth(manifest, gt) {
    var stage = $("#gt-stage");
    if (!stage || !gt) return;
    var video = $("#gt-video"), canvas = $("#gt-overlay");
    var ch = (manifest.channels || {}).gcamp_noisy;
    if (!ch) { stage.parentNode.parentNode.remove(); return; }
    setSources(video, ch.webm, ch.mp4, ch.poster);
    gateVideo(video);

    var cells = gt.cells || [];
    var peaks = cells.map(function (c) { return c.peak_dff; }).sort(function (x, y) { return x - y; });
    var maxPeak = peaks.length ? peaks[peaks.length - 1] : 1;
    var slider = $("#gt-thresh"), valOut = $("#gt-thresh-val");
    var readout = $("#gt-readout"), showBox = $("#gt-show");
    var selected = -1;
    var thresh = 0;

    function sizeCanvas() {
      var r = stage.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(r.width * dpr);
      canvas.height = Math.round(r.height * dpr);
      draw();
    }

    function draw() {
      var g = canvas.getContext("2d");
      g.clearRect(0, 0, canvas.width, canvas.height);
      if (!showBox.checked) return;
      var sx = canvas.width / gt.movie.width;
      var sy = canvas.height / gt.movie.height;
      // 500+ rings at full weight is a mesh, not an overlay. Below-threshold
      // cells are drawn as bare dots so the count stays honest without hiding
      // the movie they are supposed to be annotating.
      var r = Math.max(2.5, canvas.width / gt.movie.width * 2.1);
      var above = 0;
      for (var i = 0; i < cells.length; i++) {
        var c = cells[i];
        var on = c.peak_dff >= thresh;
        if (on) above++;
        var x = c.c * sx, y = c.r * sy;
        if (i === selected) {
          g.beginPath(); g.arc(x, y, r * 2.1, 0, 6.2832);
          g.strokeStyle = "rgba(255,180,84,.98)"; g.lineWidth = 2.4; g.stroke();
        } else if (on) {
          g.beginPath(); g.arc(x, y, r, 0, 6.2832);
          g.strokeStyle = "rgba(79,216,232,.85)"; g.lineWidth = 1.3; g.stroke();
        } else {
          g.beginPath(); g.arc(x, y, 1.2, 0, 6.2832);
          g.fillStyle = "rgba(150,168,200,.34)"; g.fill();
        }
      }
      readout.innerHTML =
        '<b>' + fmt(gt.n_neurons_total) + '</b>' +
        '<span><span class="en">neurons in the volume</span><span class="zh">体积内的神经元</span></span>' +
        '<b>' + fmt(gt.n_in_frame) + '</b>' +
        '<span><span class="en">inside this field of view</span><span class="zh">在这个视野内</span></span>' +
        '<b>' + fmt(above) + '</b>' +
        '<span><span class="en">above the threshold</span><span class="zh">高于当前阈值</span></span>';
    }

    slider.addEventListener("input", function () {
      thresh = (slider.value / 100) * maxPeak;
      valOut.textContent = thresh.toFixed(2);
      draw();
    });
    showBox.addEventListener("change", draw);
    canvas.addEventListener("click", function (e) {
      var r = canvas.getBoundingClientRect();
      var mx = (e.clientX - r.left) / r.width * gt.movie.width;
      var my = (e.clientY - r.top) / r.height * gt.movie.height;
      var best = -1, bestD = 1e9;
      for (var i = 0; i < cells.length; i++) {
        var dx = cells[i].c - mx, dy = cells[i].r - my;
        var d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      selected = bestD < 64 ? best : -1;
      draw();
      if (selected >= 0) window.dispatchEvent(
        new CustomEvent("calcia:select", { detail: cells[selected].i }));
    });

    var v = gt.verification || {};
    $("#gt-verify").innerHTML =
      '<span class="en">Overlay checked, not assumed: each dot’s pixel time course was correlated against that neuron’s own trace. Median r = ' +
      (v.median_pixel_trace_r || 0).toFixed(3) + ' with the mapping used, versus ' +
      (gt.axis_scores && gt.axis_scores["row=y"] ? gt.axis_scores["row=y"][0].toFixed(3) : "—") +
      ' for the transposed one.</span>' +
      '<span class="zh">叠加是验证过的，不是假设的：每个点所在像素的时间曲线与该神经元自身的真值曲线做了相关。当前映射下相关系数中位数 r = ' +
      (v.median_pixel_trace_r || 0).toFixed(3) + '，而转置映射只有 ' +
      (gt.axis_scores && gt.axis_scores["row=y"] ? gt.axis_scores["row=y"][0].toFixed(3) : "—") + '。</span>';

    window.addEventListener("resize", sizeCanvas);
    thresh = (slider.value / 100) * maxPeak;
    valOut.textContent = thresh.toFixed(2);
    sizeCanvas();
  }

  /* ---------------------------------------------------------------- traces
     Two canvases: the traces are drawn once, and only a one-pixel playhead is
     repainted while the movie runs. */
  function initTraces(manifest, traces) {
    var stat = $("#traces-static"), play = $("#traces-play");
    if (!stat || !traces || !traces.cells.length) return;
    var cells = traces.cells;
    var n = cells.length, T = traces.n_frames;
    var video = $("#gt-video");
    var selected = 0;

    function layout() {
      var wrap = stat.parentNode;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = wrap.clientWidth, h = Math.max(300, Math.min(560, n * 15));
      [stat, play].forEach(function (c) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
        c.style.height = h + "px";
      });
      drawStatic();
      drawPlay();
    }

    function drawStatic() {
      var g = stat.getContext("2d");
      var W = stat.width, H = stat.height;
      var padL = W * 0.055, padR = W * 0.02;
      var lane = H / n;
      g.clearRect(0, 0, W, H);
      for (var i = 0; i < n; i++) {
        var q = cells[i].q;
        var y0 = lane * (i + 0.92);
        var amp = lane * 0.86;
        g.beginPath();
        for (var t = 0; t < T; t++) {
          var x = padL + (W - padL - padR) * (t / (T - 1));
          var y = y0 - (q[t] / 255) * amp;
          if (t === 0) g.moveTo(x, y); else g.lineTo(x, y);
        }
        g.lineWidth = i === selected ? 2.1 : 1.15;
        g.strokeStyle = i === selected ? "rgba(255,180,84,.98)"
                                       : "rgba(79,216,232," + (0.30 + 0.4 * (1 - i / n)) + ")";
        g.stroke();
        var dpr = stat.clientWidth ? stat.width / stat.clientWidth : 1;
        g.font = (11 * dpr) + "px ui-monospace, monospace";
        g.fillStyle = i === selected ? "rgba(255,180,84,.9)" : "rgba(91,106,134,.85)";
        g.fillText("#" + cells[i].i, 6, y0 - 2);
      }
    }

    // The movie that owns the playhead lives in the section above, so by the
    // time these traces are on screen it is off screen and paused. Follow it
    // when it really is running, and otherwise keep our own clock at the
    // acquisition rate -- the playhead stays meaningful for free.
    var period = T / (traces.fps || 20);
    var t0 = (window.performance || Date).now();
    function playheadFrac() {
      if (video && !video.paused && video.duration) {
        return video.currentTime / video.duration;
      }
      var now = (window.performance || Date).now();
      return (((now - t0) / 1000) % period) / period;
    }

    function drawPlay() {
      var g = play.getContext("2d");
      var W = play.width, H = play.height;
      g.clearRect(0, 0, W, H);
      var padL = W * 0.055, padR = W * 0.02;
      var x = padL + (W - padL - padR) * Math.max(0, Math.min(1, playheadFrac()));
      g.strokeStyle = "rgba(232,238,251,.55)";
      g.lineWidth = 1.4;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
    }

    // Even a one-pixel repaint should not run while nobody is looking at it.
    var visible = false, raf = 0;
    function loop() {
      drawPlay();
      raf = visible ? requestAnimationFrame(loop) : 0;
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible && !raf) { t0 = (window.performance || Date).now(); loop(); }
      }, { threshold: 0.05 }).observe(stat.parentNode);
    } else {
      visible = true; loop();
    }

    window.addEventListener("calcia:select", function (e) {
      for (var i = 0; i < n; i++) {
        if (cells[i].i === e.detail) { selected = i; drawStatic(); return; }
      }
    });
    window.addEventListener("resize", layout);
    layout();
  }

  /* ------------------------------------------------------------ misc panels */
  function initPipeline(manifest) {
    var chans = manifest.channels || {};
    if (manifest.optics && manifest.optics.png) {
      $("#psf-img").src = A + manifest.optics.png;
    } else {
      var fig = $(".psf"); if (fig) fig.remove();
    }
    [["#v-clean", chans.gcamp_clean], ["#v-noisy", chans.gcamp_noisy]]
      .forEach(function (pair) {
        var v = $(pair[0]);
        if (!v) return;
        if (!pair[1]) { v.parentNode.remove(); return; }
        setSources(v, pair[1].webm, pair[1].mp4, pair[1].poster);
        gateVideo(v);
      });
  }

  function initSeries(manifest) {
    var v = $("#series-video");
    if (!v) return;
    if (!manifest.series) { $("#series-band").remove(); return; }
    setSources(v, manifest.series.webm, manifest.series.mp4,
               manifest.series.contact);
    gateVideo(v);
  }

  function initFoot(manifest, growth) {
    var note = $("#foot-note");
    if (!note) return;
    var run = manifest.hero_run || "";
    note.innerHTML =
      '<span class="en">Imaging footage from <code>' + run +
      '</code> · ' + (manifest.fps || 20) + ' fps. Growth scene: ' +
      (growth ? fmt(growth.counts.segments) + ' dendrite segments over ' +
        growth.n_iters + ' growth iterations, ' +
        Math.round((growth.counts.segments * 10 * 4) / 1024) + ' KB of geometry'
              : '—') + '.</span>' +
      '<span class="zh">成像素材来自 <code>' + run + '</code> · ' +
      (manifest.fps || 20) + ' fps。生长场景：' +
      (growth ? fmt(growth.counts.segments) + ' 段树突、' + growth.n_iters +
        ' 次生长迭代，几何数据约 ' +
        Math.round((growth.counts.segments * 10 * 4) / 1024) + ' KB' : '—') + '。</span>';
  }

  /* ------------------------------------------------------------------- boot */
  initLang();

  Promise.all([
    get(A + "manifest.json", "manifest").catch(function () { return null; }),
    get(A + "growth/growth.json", "growth").catch(function () { return null; })
  ]).then(function (res) {
    var manifest = res[0], growth = res[1];
    initHero(growth);
    if (!manifest) {
      $("#offline-note").hidden = false;
      ["#compare-band", "#gt-band", "#traces-band", "#series-band"]
        .forEach(function (s) { var n = $(s); if (n) n.remove(); });
      return;
    }
    initCompare(manifest);
    initPipeline(manifest);
    initSeries(manifest);
    initFoot(manifest, growth);
    return Promise.all([
      manifest.ground_truth ? get(A + manifest.ground_truth.json, "gt") : null,
      manifest.traces ? get(A + manifest.traces.json, "traces") : null
    ]).then(function (d) {
      initGroundTruth(manifest, d[0]);
      initTraces(manifest, d[1]);
    });
  }).catch(function (err) {
    console.error(err);
    var n = $("#offline-note"); if (n) n.hidden = false;
  });
})(window);
