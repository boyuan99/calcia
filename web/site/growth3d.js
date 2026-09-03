/* ==========================================================================
   growth3d.js -- the neuron growth animation, rendered live in the browser.

   No library. ~330 lines of WebGL 1, which is smaller than any 3D framework's
   loading spinner.

   THE REASON THIS IS CHEAP
   Growth is not animated on the CPU. Every dendrite segment is uploaded once
   with the growth iteration it was born on baked into its vertices; the vertex
   shader decides, per frame, which segments exist yet, extends the single one
   that is currently growing, and fades the rest by age. So the per-frame work
   is: write one float (uGrow), issue four draw calls. Nothing is rebuilt, no
   geometry is touched, no array is walked.

   Blending is additive with the depth test off. That is both the correct look
   -- fluorescence really does add -- and the reason no sorting is ever needed.
   Depth is carried by fog instead of by occlusion.
   ========================================================================== */
(function (global) {
  "use strict";

  var VERT_RIBBON = [
    "attribute vec3 aP0, aP1, aColor;",
    "attribute vec2 aQuad, aRad;",     // (side, t), (r0, r1)
    "attribute float aBirth;",
    "uniform mat4 uMVP;",
    "uniform vec2 uAspect;",
    "uniform float uGrow, uFocal, uMinPx, uAgeSpan, uMature;",
    "varying vec3 vColor;",
    "varying float vDepth, vEdge, vAge;",
    "void main() {",
    "  if (uGrow < aBirth) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }",
    "  float f = clamp(uGrow - aBirth, 0.0, 1.0);",   // extension of this edge
    "  vec3 tip = mix(aP0, aP1, f);",
    "  float rTip = mix(aRad.x, aRad.y, f);",
    "  vec4 c0 = uMVP * vec4(aP0, 1.0);",
    "  vec4 c1 = uMVP * vec4(tip, 1.0);",
    "  vec4 c  = mix(c0, c1, aQuad.y);",
    "  vec2 s0 = c0.xy / max(c0.w, 1e-4);",
    "  vec2 s1 = c1.xy / max(c1.w, 1e-4);",
    "  vec2 d  = (s1 - s0) * uAspect;",
    "  vec2 dir = (length(d) < 1e-7) ? vec2(1.0, 0.0) : normalize(d);",
    "  vec2 nrm = vec2(-dir.y, dir.x) / uAspect;",
    "  float age = clamp((uGrow - aBirth) / uAgeSpan, 0.0, 1.0);",
    "  float mature = uMature + (1.0 - uMature) * age;",
    "  float r = mix(aRad.x, rTip, aQuad.y) * mature;",
    "  float halfW = max(r * uFocal / max(c.w, 1e-4), uMinPx);",   // 'half' is a reserved GLSL word
    "  gl_Position = vec4(c.xy + nrm * aQuad.x * halfW * c.w, c.z, c.w);",
    "  vColor = aColor; vDepth = c.w; vEdge = aQuad.x; vAge = age;",
    "}"
  ].join("\n");

  var FRAG_RIBBON = [
    "precision mediump float;",
    "varying vec3 vColor;",
    "varying float vDepth, vEdge, vAge;",
    "uniform float uFogNear, uFogFar, uFog, uAlpha;",
    "void main() {",
    "  float e = abs(vEdge);",
    "  float cover = smoothstep(1.0, 0.55, e);",       // soft antialiased edge
    "  float lit = 1.0 - 0.42 * e * e;",               // fake cylinder shading
    // A tip that was born a moment ago is white-hot and cools into its own hue.
    "  vec3 col = mix(vec3(1.0), vColor, smoothstep(0.0, 1.0, vAge)) * lit;",
    "  float fog = 1.0 - uFog * smoothstep(uFogNear, uFogFar, vDepth);",
    "  gl_FragColor = vec4(col * fog * cover * uAlpha, 1.0);",
    "}"
  ].join("\n");

  var VERT_POINT = [
    "attribute vec3 aPos, aColor;",
    "attribute vec2 aParam;",          // (birth or kill, base size)
    "uniform mat4 uMVP;",
    "uniform float uGrow, uFocal, uMode, uAgeSpan, uSizeScale;",
    "varying vec3 vColor;",
    "varying float vDepth, vFade;",
    "void main() {",
    "  float k = aParam.x;",
    "  float show = 1.0;",
    // mode 0: attractor dust, alive until it is consumed
    // mode 1: growth front, a spark that fades over uAgeSpan iterations
    // mode 2: soma, always on
    "  if (uMode < 0.5) { show = (k < 0.0 || uGrow < k) ? 1.0 : 0.0; }",
    "  else if (uMode < 1.5) {",
    "    show = (uGrow >= k) ? 1.0 - clamp((uGrow - k) / uAgeSpan, 0.0, 1.0) : 0.0;",
    "  }",
    "  if (show <= 0.001) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }",
    "  vec4 c = uMVP * vec4(aPos, 1.0);",
    "  gl_Position = c;",
    "  gl_PointSize = clamp(aParam.y * uSizeScale * uFocal / max(c.w, 1e-4),",
    "                       1.0, 220.0);",
    "  vColor = aColor; vDepth = c.w; vFade = show;",
    "}"
  ].join("\n");

  var FRAG_POINT = [
    "precision mediump float;",
    "varying vec3 vColor;",
    "varying float vDepth, vFade;",
    "uniform float uFogNear, uFogFar, uFog, uAlpha, uCore;",
    "void main() {",
    "  float d = length(gl_PointCoord - 0.5) * 2.0;",
    "  if (d > 1.0) discard;",
    "  float a = pow(1.0 - d, uCore);",
    "  float fog = 1.0 - uFog * smoothstep(uFogNear, uFogFar, vDepth);",
    "  gl_FragColor = vec4(vColor * a * fog * vFade * uAlpha, 1.0);",
    "}"
  ].join("\n");

  var VERT_QUAD = [
    "attribute vec2 aXY;",
    "varying vec2 vUV;",
    "void main() { vUV = aXY * 0.5 + 0.5; gl_Position = vec4(aXY, 0.0, 1.0); }"
  ].join("\n");

  var FRAG_BRIGHT = [
    "precision mediump float;",
    "varying vec2 vUV;",
    "uniform sampler2D uTex;",
    "uniform float uThreshold;",
    "void main() {",
    "  vec3 c = texture2D(uTex, vUV).rgb;",
    "  gl_FragColor = vec4(max(c - uThreshold, 0.0) / max(1.0 - uThreshold, 1e-3), 1.0);",
    "}"
  ].join("\n");

  var FRAG_BLUR = [
    "precision mediump float;",
    "varying vec2 vUV;",
    "uniform sampler2D uTex;",
    "uniform vec2 uStep;",
    "void main() {",
    "  vec3 s = texture2D(uTex, vUV).rgb * 0.227;",
    "  s += (texture2D(uTex, vUV + uStep * 1.385).rgb",
    "      + texture2D(uTex, vUV - uStep * 1.385).rgb) * 0.316;",
    "  s += (texture2D(uTex, vUV + uStep * 3.231).rgb",
    "      + texture2D(uTex, vUV - uStep * 3.231).rgb) * 0.070;",
    "  gl_FragColor = vec4(s, 1.0);",
    "}"
  ].join("\n");

  var FRAG_COMPOSITE = [
    "precision mediump float;",
    "varying vec2 vUV;",
    "uniform sampler2D uScene, uBloom;",
    "uniform float uStrength, uVignette;",
    "void main() {",
    "  vec3 c = texture2D(uScene, vUV).rgb + texture2D(uBloom, vUV).rgb * uStrength;",
    "  c = c / (1.0 + 0.26 * c);",                      // filmic shoulder
    "  vec2 p = vUV * 2.0 - 1.0;",
    "  c *= 1.0 - uVignette * pow(dot(p, p) * 0.5, 1.1);",
    "  gl_FragColor = vec4(c, 1.0);",
    "}"
  ].join("\n");

  // Camera choreography. A full turn takes ~39 s and the elevation sweep ~57 s;
  // the two periods do not divide each other, so the view never repeats exactly.
  var SPIN_RATE = 0.27;        // rad/s of azimuth -- a turn every ~23 s
  var EL_RATE = 0.13;          // rad/s of the elevation oscillator (~48 s)
  var EL_AMPLITUDE = 0.30;     // rad, about 17 degrees either side
  var RESUME_DELAY = 2.2;      // s of stillness before auto motion returns
  var RESUME_EASE = 2.0;       // s to ease it back to full

  // ------------------------------------------------------------------ gl util
  function shader(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s) + "\n" + src);
    }
    return s;
  }
  function program(gl, vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, shader(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, shader(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p));
    }
    p.at = {};
    p.un = {};
    var i, n = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES);
    for (i = 0; i < n; i++) {
      var a = gl.getActiveAttrib(p, i).name;
      p.at[a] = gl.getAttribLocation(p, a);
    }
    n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (i = 0; i < n; i++) {
      var u = gl.getActiveUniform(p, i).name.replace("[0]", "");
      p.un[u] = gl.getUniformLocation(p, u);
    }
    return p;
  }
  function buffer(gl, data, target) {
    var b = gl.createBuffer();
    target = target || gl.ARRAY_BUFFER;
    gl.bindBuffer(target, b);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return b;
  }
  function target(gl, w, h) {
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA,
                  gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, tex, 0);
    var ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return ok ? { tex: tex, fb: fb, w: w, h: h } : null;
  }

  // ----------------------------------------------------------------- matrices
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0];
  }
  function lookAt(eye, at, up) {
    var z = sub(eye, at); norm3(z);
    var x = cross(up, z); norm3(x);
    var y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
            -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
  }
  function mul(a, b) {
    var o = new Array(16), i, j, k;
    for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
      var s = 0;
      for (k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
    return o;
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]];
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function norm3(v) {
    var l = Math.sqrt(dot(v, v)) || 1;
    v[0] /= l; v[1] /= l; v[2] /= l;
  }

  // ------------------------------------------------------------ geometry prep
  function ribbonBuffers(gl, p0, p1, r0, r1, birth, colors) {
    var n = birth.length;
    var V = new Float32Array(n * 4 * 14);
    var I = new Uint16Array(n * 6);
    var quad = [[-1, 0], [1, 0], [-1, 1], [1, 1]];
    var v = 0, ii = 0;
    for (var s = 0; s < n; s++) {
      var base = s * 4;
      for (var q = 0; q < 4; q++) {
        V[v++] = p0[s * 3]; V[v++] = p0[s * 3 + 1]; V[v++] = p0[s * 3 + 2];
        V[v++] = p1[s * 3]; V[v++] = p1[s * 3 + 1]; V[v++] = p1[s * 3 + 2];
        V[v++] = colors[s * 3]; V[v++] = colors[s * 3 + 1]; V[v++] = colors[s * 3 + 2];
        V[v++] = quad[q][0]; V[v++] = quad[q][1];
        V[v++] = r0[s]; V[v++] = r1[s];
        V[v++] = birth[s];
      }
      I[ii++] = base; I[ii++] = base + 1; I[ii++] = base + 2;
      I[ii++] = base + 2; I[ii++] = base + 1; I[ii++] = base + 3;
    }
    return {
      vbo: buffer(gl, V), ibo: buffer(gl, I, gl.ELEMENT_ARRAY_BUFFER),
      count: n * 6
    };
  }

  function bindRibbon(gl, prog, buf) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.vbo);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf.ibo);
    var S = 14 * 4;
    var map = [["aP0", 3, 0], ["aP1", 3, 12], ["aColor", 3, 24],
               ["aQuad", 2, 36], ["aRad", 2, 44], ["aBirth", 1, 52]];
    for (var i = 0; i < map.length; i++) {
      var loc = prog.at[map[i][0]];
      if (loc === undefined || loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, map[i][1], gl.FLOAT, false, S, map[i][2]);
    }
  }

  function pointBuffer(gl, pos, param, colors) {
    var n = param.length / 2;
    var V = new Float32Array(n * 8);
    for (var i = 0; i < n; i++) {
      V[i * 8] = pos[i * 3]; V[i * 8 + 1] = pos[i * 3 + 1]; V[i * 8 + 2] = pos[i * 3 + 2];
      V[i * 8 + 3] = colors[i * 3]; V[i * 8 + 4] = colors[i * 3 + 1];
      V[i * 8 + 5] = colors[i * 3 + 2];
      V[i * 8 + 6] = param[i * 2]; V[i * 8 + 7] = param[i * 2 + 1];
    }
    return { vbo: buffer(gl, V), count: n };
  }

  function bindPoints(gl, prog, buf) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.vbo);
    var S = 8 * 4;
    var map = [["aPos", 3, 0], ["aColor", 3, 12], ["aParam", 2, 24]];
    for (var i = 0; i < map.length; i++) {
      var loc = prog.at[map[i][0]];
      if (loc === undefined || loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, map[i][1], gl.FLOAT, false, S, map[i][2]);
    }
  }

  // =========================================================== the renderer
  function GrowthScene(canvas, meta, blob, opts) {
    opts = opts || {};
    var gl = canvas.getContext("webgl", {
      alpha: false, antialias: false, depth: false,
      powerPreference: "low-power", preserveDrawingBuffer: false
    }) || canvas.getContext("experimental-webgl");
    if (!gl) throw new Error("WebGL unavailable");

    this.gl = gl;
    this.canvas = canvas;
    this.meta = meta;
    this.opts = opts;
    this.bloom = opts.bloom !== false;
    this.maxDpr = opts.maxDpr || 1.5;

    var A = meta.arrays;
    function get(name) {
      var a = A[name];
      return new Float32Array(blob, a.offset * 4, a.length);
    }
    var pal = meta.palette.map(function (c) {
      return [c[0] / 255, c[1] / 255, c[2] / 255];
    });

    // ---- dendrites
    var segNid = get("seg_nid"), nSeg = segNid.length;
    var segCol = new Float32Array(nSeg * 3);
    for (var i = 0; i < nSeg; i++) {
      var c = pal[segNid[i] | 0] || [0.5, 0.8, 1];
      segCol[i * 3] = c[0]; segCol[i * 3 + 1] = c[1]; segCol[i * 3 + 2] = c[2];
    }
    var segP0 = get("seg_p0"), segP1 = get("seg_p1"), segBirth = get("seg_birth");
    this.prog = program(gl, VERT_RIBBON, FRAG_RIBBON);
    this.progPt = program(gl, VERT_POINT, FRAG_POINT);
    this.dend = ribbonBuffers(gl, segP0, segP1, get("seg_r0"), get("seg_r1"),
                              segBirth, segCol);

    // ---- background wisps: one flat colour, born before anything else
    var wp0 = get("wisp_p0"), wp1 = get("wisp_p1");
    var nW = wp0.length / 3;
    var wr = new Float32Array(nW), wb = new Float32Array(nW),
        wc = new Float32Array(nW * 3);
    for (i = 0; i < nW; i++) {
      wr[i] = 0.0016; wb[i] = -1;
      wc[i * 3] = 0.30; wc[i * 3 + 1] = 0.40; wc[i * 3 + 2] = 0.62;
    }
    this.wisp = ribbonBuffers(gl, wp0, wp1, wr, wr, wb, wc);

    // ---- attractor dust
    var kill = get("attr_kill"), nA = kill.length;
    var aParam = new Float32Array(nA * 2), aCol = new Float32Array(nA * 3);
    for (i = 0; i < nA; i++) {
      aParam[i * 2] = kill[i]; aParam[i * 2 + 1] = 0.010;
      aCol[i * 3] = 0.58; aCol[i * 3 + 1] = 0.72; aCol[i * 3 + 2] = 0.98;
    }
    this.dust = pointBuffer(gl, get("attr_pos"), aParam, aCol);

    // ---- growth front sparks: one per segment, at the tip it just reached
    var sParam = new Float32Array(nSeg * 2), sCol = new Float32Array(nSeg * 3);
    for (i = 0; i < nSeg; i++) {
      sParam[i * 2] = segBirth[i]; sParam[i * 2 + 1] = 0.016;
      sCol[i * 3] = 1; sCol[i * 3 + 1] = 1; sCol[i * 3 + 2] = 1;
    }
    this.spark = pointBuffer(gl, segP1, sParam, sCol);

    // ---- somata (re-uploaded each frame: 4 points, nothing)
    this.somaPos = get("soma_pos");
    this.somaR = get("soma_r");
    this.nSoma = this.somaR.length;
    this.somaPal = pal;
    this.somaData = new Float32Array(this.nSoma * 8);
    this.somaBuf = { vbo: gl.createBuffer(), count: this.nSoma };

    this.quad = buffer(gl, new Float32Array([-1, -1, 3, -1, -1, 3]));
    if (this.bloom) {
      try {
        this.pBright = program(gl, VERT_QUAD, FRAG_BRIGHT);
        this.pBlur = program(gl, VERT_QUAD, FRAG_BLUR);
        this.pComp = program(gl, VERT_QUAD, FRAG_COMPOSITE);
      } catch (e) { this.bloom = false; }
    }

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);          // additive: fluorescence adds

    // ---- camera / clock state
    // The camera is choreographed rather than parked: azimuth turns
    // continuously and elevation breathes on a slower, coprime period, so the
    // scene is shown from a different angle every time you look at it, and a
    // dendritic tree -- which is mostly empty space -- reads as a 3D object
    // instead of a flat tangle. Both run on their own clock, not the growth
    // loop, so they keep working while it is paused or scrubbed.
    this.az = -0.7; this.el = 0.30; this.dist = 2.75;
    this.azVel = 0;
    this.elPhase = 0;
    this.autoSpin = true;
    this.idle = 99;                    // seconds since the last drag
    this.playing = true;
    this.time = 0;
    this.scrub = null;
    this.fbo = null;
    this.resize();
  }

  // 0 while you are dragging and for a moment after, then eases back to 1.
  //
  // `autoSpin` is only ever set false by an explicit request to hold still
  // (reduced motion). Nothing else may latch it off -- an earlier version let a
  // single drag switch the orbit off for the rest of the session, which is
  // indistinguishable from the feature not existing.
  GrowthScene.prototype.autoAmount = function () {
    if (!this.autoSpin) return 0;
    return Math.max(0, Math.min(1, (this.idle - RESUME_DELAY) / RESUME_EASE));
  };

  GrowthScene.prototype.resize = function () {
    var gl = this.gl, c = this.canvas;
    var dpr = Math.min(global.devicePixelRatio || 1, this.maxDpr);
    var w = Math.max(1, Math.round(c.clientWidth * dpr));
    var h = Math.max(1, Math.round(c.clientHeight * dpr));
    // Bound total fill rate regardless of how big the element gets.
    var cap = this.opts.maxPixels || 2200000;
    var over = (w * h) / cap;
    if (over > 1) { var k = Math.sqrt(over); w = Math.round(w / k); h = Math.round(h / k); }
    // Ignore small changes. Reallocating three framebuffers is expensive, and a
    // scrollbar appearing or a caption reflowing by a few pixels is not worth a
    // GPU stall. Anything larger is rebuilt at most a few times a second.
    if (this.fbo && Math.abs(w - c.width) < 24 && Math.abs(h - c.height) < 24) return;
    var now = (global.performance || Date).now();
    if (this.fbo && now - (this._lastResize || 0) < 220) return;
    this._lastResize = now;

    c.width = w; c.height = h;
    if (this.bloom) {
      // The old targets must be released explicitly -- WebGL will not collect
      // them, and a page that resizes repeatedly would leak a full-resolution
      // texture every time.
      this.freeTargets();
      var qw = Math.max(1, w >> 2), qh = Math.max(1, h >> 2);
      this.fbo = target(gl, w, h);
      this.fboA = target(gl, qw, qh);
      this.fboB = target(gl, qw, qh);
      if (!this.fbo || !this.fboA || !this.fboB) { this.bloom = false; this.fbo = null; }
    }
  };

  GrowthScene.prototype.freeTargets = function () {
    var gl = this.gl;
    [this.fbo, this.fboA, this.fboB].forEach(function (t) {
      if (!t) return;
      gl.deleteTexture(t.tex);
      gl.deleteFramebuffer(t.fb);
    });
    this.fbo = this.fboA = this.fboB = null;
  };

  GrowthScene.prototype.beat = function (t01, name) {
    var b = this.meta.beats[name];
    if (!b) return 0;
    return Math.max(0, Math.min(1, (t01 - b.start) / Math.max(b.end - b.start, 1e-6)));
  };

  GrowthScene.prototype.drawQuad = function (prog) {
    var gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(prog.at.aXY);
    gl.vertexAttribPointer(prog.at.aXY, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  GrowthScene.prototype.render = function (t01) {
    var gl = this.gl, W = this.canvas.width, H = this.canvas.height;
    var smooth = function (x) { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };

    var grow01 = 0.5 - 0.5 * Math.cos(Math.PI * this.beat(t01, "grow"));
    var uGrow = grow01 * this.meta.n_iters;

    // camera
    var fov = 0.50;                                   // ~28 degrees
    var auto = this.autoAmount();
    var el = Math.max(-1.25, Math.min(1.25,
      this.el + EL_AMPLITUDE * Math.sin(this.elPhase) * auto));
    // Distance is CONSTANT. A dolly tied to the growth clock made the scene
    // pulse in size every loop, which reads as jank rather than as camera work;
    // rotation carries the motion instead.
    var dist = this.dist;
    var eye = [Math.cos(el) * Math.cos(this.az) * dist,
               Math.cos(el) * Math.sin(this.az) * dist,
               Math.sin(el) * dist];
    var aspect = W / H;
    var mvp = mul(perspective(fov, aspect, 0.05, 40), lookAt(eye, [0, 0, 0], [0, 0, 1]));
    var focal = 1 / Math.tan(fov / 2);

    var fbTarget = this.bloom ? this.fbo.fb : null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbTarget);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.017, 0.024, 0.041, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    var fogNear = dist - 1.0, fogFar = dist + 1.25;
    var p = this.prog;
    gl.useProgram(p);
    gl.uniformMatrix4fv(p.un.uMVP, false, new Float32Array(mvp));
    gl.uniform2f(p.un.uAspect, aspect, 1);
    gl.uniform1f(p.un.uFocal, focal);
    gl.uniform1f(p.un.uMinPx, 0.9 / H);
    gl.uniform1f(p.un.uAgeSpan, 5);
    gl.uniform1f(p.un.uFogNear, fogNear);
    gl.uniform1f(p.un.uFogFar, fogFar);
    gl.uniform1f(p.un.uFog, 0.72);

    // background neuropil, then the trees on top of it
    gl.uniform1f(p.un.uGrow, 1e9);
    gl.uniform1f(p.un.uMature, 1);
    gl.uniform1f(p.un.uAlpha, 0.30 * smooth(this.beat(t01, "axons_in")));
    if (p.un.uAlpha && this.wisp.count) {
      bindRibbon(gl, p, this.wisp);
      gl.drawElements(gl.TRIANGLES, this.wisp.count, gl.UNSIGNED_SHORT, 0);
    }
    gl.uniform1f(p.un.uGrow, uGrow);
    gl.uniform1f(p.un.uMature, 0.32);
    gl.uniform1f(p.un.uAlpha, 1);
    bindRibbon(gl, p, this.dend);
    gl.drawElements(gl.TRIANGLES, this.dend.count, gl.UNSIGNED_SHORT, 0);

    // points: dust, sparks, somata
    var q = this.progPt;
    gl.useProgram(q);
    gl.uniformMatrix4fv(q.un.uMVP, false, new Float32Array(mvp));
    gl.uniform1f(q.un.uFocal, focal);
    gl.uniform1f(q.un.uGrow, uGrow);
    gl.uniform1f(q.un.uAgeSpan, 4);
    gl.uniform1f(q.un.uFogNear, fogNear);
    gl.uniform1f(q.un.uFogFar, fogFar);
    gl.uniform1f(q.un.uFog, 0.72);
    gl.uniform1f(q.un.uSizeScale, H * 0.5);

    var dustA = 0.85 * smooth(this.beat(t01, "dust_in"))
              * (1 - 0.55 * smooth(this.beat(t01, "grow")));
    if (dustA > 0.01) {
      gl.uniform1f(q.un.uMode, 0);
      gl.uniform1f(q.un.uCore, 1.6);
      gl.uniform1f(q.un.uAlpha, dustA);
      bindPoints(gl, q, this.dust);
      gl.drawArrays(gl.POINTS, 0, this.dust.count);
    }

    var g = this.beat(t01, "grow");
    if (g > 0 && g < 1) {
      gl.uniform1f(q.un.uMode, 1);
      gl.uniform1f(q.un.uCore, 2.2);
      gl.uniform1f(q.un.uAlpha, 0.95);
      bindPoints(gl, q, this.spark);
      gl.drawArrays(gl.POINTS, 0, this.spark.count);
    }

    this.drawSomata(t01, q);

    if (this.bloom) this.composite(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  GrowthScene.prototype.drawSomata = function (t01, q) {
    var gl = this.gl;
    var bl = this.beat(t01, "bloom");
    var pulse = this.beat(t01, "pulse");
    var D = this.somaData;
    for (var i = 0; i < this.nSoma; i++) {
      var stagger = this.nSoma > 1 ? (i / (this.nSoma - 1)) * 0.55 : 0;
      var local = Math.max(0, Math.min(1, (bl - stagger) / 0.45));
      var grow = local * local * (3 - 2 * local);
      // A calcium-like flash: fast rise, slow decay, offset per cell. The film
      // uses real burst spikes; here four staggered transients read the same at
      // 20x less machinery, and the honest version is one section further down.
      var ph = (pulse * 3.1 + i * 0.37) % 1.0;
      var flash = pulse > 0 ? Math.exp(-ph * 4.2) * (1 - Math.exp(-ph * 40)) : 0;
      var col = this.somaPal[i] || [0.5, 0.9, 1];
      var k = i * 8;
      D[k] = this.somaPos[i * 3];
      D[k + 1] = this.somaPos[i * 3 + 1];
      D[k + 2] = this.somaPos[i * 3 + 2];
      D[k + 3] = col[0] * (1 + 1.6 * flash);
      D[k + 4] = col[1] * (1 + 1.6 * flash);
      D[k + 5] = col[2] * (1 + 1.6 * flash);
      D[k + 6] = -1;
      D[k + 7] = this.somaR[i] * (0.35 + 0.65 * grow) * 2.0;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.somaBuf.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, D, gl.DYNAMIC_DRAW);
    gl.uniform1f(q.un.uMode, 2);
    bindPoints(gl, q, this.somaBuf);
    gl.uniform1f(q.un.uCore, 0.85);
    gl.uniform1f(q.un.uAlpha, 0.95);
    gl.drawArrays(gl.POINTS, 0, this.nSoma);
    // a wider, softer copy: the halo the film got from its bloom pass
    gl.uniform1f(q.un.uCore, 2.6);
    gl.uniform1f(q.un.uAlpha, 0.42);
    gl.drawArrays(gl.POINTS, 0, this.nSoma);
  };

  GrowthScene.prototype.composite = function (W, H) {
    var gl = this.gl, qw = this.fboA.w, qh = this.fboA.h;
    gl.blendFunc(gl.ONE, gl.ZERO);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboA.fb);
    gl.viewport(0, 0, qw, qh);
    gl.useProgram(this.pBright);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fbo.tex);
    gl.uniform1i(this.pBright.un.uTex, 0);
    gl.uniform1f(this.pBright.un.uThreshold, 0.36);
    this.drawQuad(this.pBright);

    gl.useProgram(this.pBlur);
    gl.uniform1i(this.pBlur.un.uTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboB.fb);
    gl.bindTexture(gl.TEXTURE_2D, this.fboA.tex);
    gl.uniform2f(this.pBlur.un.uStep, 1 / qw, 0);
    this.drawQuad(this.pBlur);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboA.fb);
    gl.bindTexture(gl.TEXTURE_2D, this.fboB.tex);
    gl.uniform2f(this.pBlur.un.uStep, 0, 1 / qh);
    this.drawQuad(this.pBlur);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.pComp);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fbo.tex);
    gl.uniform1i(this.pComp.un.uScene, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.fboA.tex);
    gl.uniform1i(this.pComp.un.uBloom, 1);
    gl.uniform1f(this.pComp.un.uStrength, 1.15);
    gl.uniform1f(this.pComp.un.uVignette, 0.34);
    this.drawQuad(this.pComp);
    gl.activeTexture(gl.TEXTURE0);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  };

  // ------------------------------------------------------------------ driver
  GrowthScene.prototype.attach = function (handlers) {
    var self = this, c = this.canvas, last = 0, raf = 0, visible = false;
    var drag = false, lastX = 0, lastY = 0;

    function frame(now) {
      var dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
      last = now;
      if (self.playing && self.scrub === null) {
        self.time = (self.time + dt / self.meta.duration_s) % 1;
      }
      // Auto motion fades back in a couple of seconds after you let go, so
      // dragging never feels like it is fighting the camera.
      self.idle = drag ? 0 : self.idle + dt;
      var auto = self.autoAmount();
      self.az += dt * SPIN_RATE * auto;
      self.elPhase += dt * EL_RATE * auto;
      self.az += self.azVel * dt;
      self.azVel *= Math.pow(0.02, dt);
      var t = self.scrub === null ? self.time : self.scrub;
      self.resize();
      self.render(t);
      if (handlers && handlers.onTime) handlers.onTime(t);
      raf = visible ? global.requestAnimationFrame(frame) : 0;
    }
    function start() { if (!raf) { last = 0; raf = global.requestAnimationFrame(frame); } }

    // Paint once immediately. The observer below is asynchronous, and a hero
    // that is blank until the first intersection callback reads as broken.
    this.resize();
    this.render(this.scrub === null ? this.time : this.scrub);

    // Nothing renders while the hero is off screen or the tab is hidden.
    if ("IntersectionObserver" in global) {
      new IntersectionObserver(function (e) {
        visible = e[0].isIntersecting && !document.hidden;
        if (visible) start();
      }, { threshold: 0.02 }).observe(c);
    } else { visible = true; start(); }
    document.addEventListener("visibilitychange", function () {
      visible = !document.hidden && visible;
      if (visible) start();
    });

    c.addEventListener("pointerdown", function (e) {
      drag = true; lastX = e.clientX; lastY = e.clientY;
      self.idle = 0;
      c.setPointerCapture(e.pointerId);
      c.classList.add("is-drag");
    });
    c.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      self.az -= dx * 0.0065;
      self.el = Math.max(-1.35, Math.min(1.35, self.el + dy * 0.005));
      self.azVel = -dx * 0.06;
    });
    function endDrag() { drag = false; c.classList.remove("is-drag"); }
    c.addEventListener("pointerup", endDrag);
    c.addEventListener("pointercancel", endDrag);

    this.start = start;
    return this;
  };

  global.GrowthScene = GrowthScene;

  function b64ToBuffer(b64) {
    var bin = global.atob(b64), n = bin.length, out = new Uint8Array(n);
    for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }

  // `inline` is window.CALCIA_DATA when web/bundle_data.py has been run: the
  // same payload, already in the document, so nothing is fetched and the page
  // works from file:// with no server.
  global.loadGrowthScene = function (canvas, base, opts, inline) {
    if (inline && inline.growth && inline.growthBin) {
      return Promise.resolve({
        scene: new GrowthScene(canvas, inline.growth,
                               b64ToBuffer(inline.growthBin), opts),
        meta: inline.growth
      });
    }
    return fetch(base + "growth.json").then(function (r) { return r.json(); })
      .then(function (meta) {
        return fetch(base + meta.bin).then(function (r) { return r.arrayBuffer(); })
          .then(function (buf) {
            return { scene: new GrowthScene(canvas, meta, buf, opts), meta: meta };
          });
      });
  };
})(window);
