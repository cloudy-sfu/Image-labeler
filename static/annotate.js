/*
  Canvas-based annotation engine for Flask Image Labeler.
  Supports bbox, polygon (with vertex editing),
  keypoint/skeleton, classification, SAM magic-lasso.
  Undo/redo, right-click to close polygon.
  Ref: YOLO annotation format — https://docs.ultralytics.com/datasets/detect/
  Ref: SAM — https://github.com/facebookresearch/segment-anything
*/

(function () {
  "use strict";

  // ════════════════════════════════════════════════════════
  //  §1  CONSTANTS & CONFIG
  // ════════════════════════════════════════════════════════

  const canvas = document.getElementById("mainCanvas");
  const ctx    = canvas.getContext("2d");
  const wrap   = canvas.parentElement;

  const task          = PROJECT.task;
  const labelGroups   = PROJECT.label_groups   || [];
  const keypointNames = PROJECT.keypoint_names || [];
  const skeletonEdges = PROJECT.skeleton_edges || [];

  const MAX_UNDO     = 60;
  const DRAWING_COLOR = "#4a6cf7";
  const STICKY_KEY    = "labeler_sticky_" + PID;
  const VERTEX_GRAB_R = 8;   // px screen-space
  const KPT_HIT_R     = 10;  // px image-space (divided by zoom at use-site)
  const SEMANTIC_FILL_ALPHA = "66";

  const BASE_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324","#800000","#aaffc3","#808000","#000075",
    "#f44336","#9c27b0","#2196f3","#009688","#ff9800","#795548",
    "#607d8b","#e91e63","#00bcd4","#8bc34a","#ffc107","#673ab7"
  ];

  // ════════════════════════════════════════════════════════
  //  §2  APPLICATION STATE
  // ════════════════════════════════════════════════════════

  const state = {
    img: new Image(),
    imgW: 1, imgH: 1,
    zoom: 1, panX: 0, panY: 0,
    annotations: [],
    selectedIdx: -1,
    tool: "",
    drawing: null,
    dirty: false,
    isPanning: false,
    panStart: null,
    draggingVertex: null,

    // Undo / Redo
    undoStack: [],
    redoStack: [],

    // Sticky labels (persisted per-project in sessionStorage)
    stickyLabels: null,
  };

  // SAM / Magic Lasso state (kept separate for clarity)
  const sam = {
    available: false,
    cacheKey: null,
    embedReady: false,
    embedPolling: null,
    promptPoints: [],   // { x, y, label } in image-pixel coords
    previewPolys: null,  // array of polygons (pixel coords)
    busy: false,
  };

  // ════════════════════════════════════════════════════════
  //  §3  UTILITY: COORDINATES & GEOMETRY
  // ════════════════════════════════════════════════════════

  function screenToImage(sx, sy) {
    return [(sx - state.panX) / state.zoom,
            (sy - state.panY) / state.zoom];
  }
  function imageToScreen(ix, iy) {
    return [ix * state.zoom + state.panX,
            iy * state.zoom + state.panY];
  }
  function normToImage(nx, ny) { return [nx * state.imgW, ny * state.imgH]; }
  function imageToNorm(ix, iy) { return [ix / state.imgW, iy / state.imgH]; }

  function distPointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return Math.hypot(px - ax, py - ay);
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  function pointInPolygon(x, y, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if (((yi > y) !== (yj > y)) &&
          (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  function segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy) {
    const eps = 1e-9;
    const orient = (px, py, qx, qy, rx, ry) =>
      (qy - py) * (rx - qx) - (qx - px) * (ry - qy);

    const o1 = orient(ax, ay, bx, by, cx, cy);
    const o2 = orient(ax, ay, bx, by, dx, dy);
    const o3 = orient(cx, cy, dx, dy, ax, ay);
    const o4 = orient(cx, cy, dx, dy, bx, by);

    if ((o1 > eps && o2 < -eps || o1 < -eps && o2 > eps) &&
        (o3 > eps && o4 < -eps || o3 < -eps && o4 > eps)) return true;

    return false;
  }

  function polygonsOverlap(polyA, polyB) {
    if (!polyA.length || !polyB.length) return false;
    for (let i = 0; i < polyA.length; i++) {
      const ni = (i + 1) % polyA.length;
      const [ax, ay] = polyA[i];
      const [bx, by] = polyA[ni];
      for (let j = 0; j < polyB.length; j++) {
        const nj = (j + 1) % polyB.length;
        const [cx, cy] = polyB[j];
        const [dx, dy] = polyB[nj];
        if (segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy)) return true;
      }
    }
    return pointInPolygon(polyA[0][0], polyA[0][1], polyB) ||
           pointInPolygon(polyB[0][0], polyB[0][1], polyA);
  }

  function semanticPolygonOverlap(candidatePoly, ignoreIdx = -1) {
    if (task !== "semantic_segmentation" || !candidatePoly || candidatePoly.length < 3) return false;
    for (let i = 0; i < state.annotations.length; i++) {
      if (i === ignoreIdx) continue;
      const ann = state.annotations[i];
      if (!ann.polygon) continue;
      const poly = ann.polygon.map(([nx, ny]) => normToImage(nx, ny));
      if (polygonsOverlap(candidatePoly, poly)) return true;
    }
    return false;
  }

  function selectedSemanticOverlap() {
    if (task !== "semantic_segmentation") return false;
    const idx = state.selectedIdx;
    if (idx < 0 || idx >= state.annotations.length) return false;
    const ann = state.annotations[idx];
    if (!ann || !ann.polygon) return false;
    const candidate = ann.polygon.map(([nx, ny]) => normToImage(nx, ny));
    return semanticPolygonOverlap(candidate, idx);
  }

  /** Screen-space distance check for a handle (vertex / keypoint). */
  function isNearScreen(sx, sy, targetSx, targetSy, radius) {
    return Math.hypot(sx - targetSx, sy - targetSy) < radius;
  }

  function getMousePos(e) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  // ════════════════════════════════════════════════════════
  //  §4  UTILITY: LABEL COLORS
  // ════════════════════════════════════════════════════════

  const colorCache = {};

  function colorForLabels(labels) {
    const key = labelGroups.map(g => labels[g.name] || "").join("\x00");
    if (colorCache[key]) return colorCache[key];
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
    }
    colorCache[key] = BASE_PALETTE[Math.abs(hash) % BASE_PALETTE.length];
    return colorCache[key];
  }

  // ════════════════════════════════════════════════════════
  //  §5  STICKY LABELS
  // ════════════════════════════════════════════════════════

  function loadStickyLabels() {
    try {
      const raw = sessionStorage.getItem(STICKY_KEY);
      if (raw) { state.stickyLabels = JSON.parse(raw); return; }
    } catch (_) { /* ignore */ }
    state.stickyLabels = null;
  }

  function saveStickyLabels(labels) {
    state.stickyLabels = Object.assign({}, labels);
    try { sessionStorage.setItem(STICKY_KEY, JSON.stringify(state.stickyLabels)); } catch (_) { /* ignore */ }
  }

  function currentLabels() {
    if (state.stickyLabels) {
      const labels = {};
      labelGroups.forEach(g => {
        const names = g.labels.map(l => l.name);
        labels[g.name] = (state.stickyLabels[g.name] && names.includes(state.stickyLabels[g.name]))
          ? state.stickyLabels[g.name]
          : (names[0] || "");
      });
      return labels;
    }
    return defaultLabels();
  }

  function defaultLabels() {
    const labels = {};
    labelGroups.forEach(g => { labels[g.name] = g.labels[0].name; });
    return labels;
  }

  // ════════════════════════════════════════════════════════
  //  §6  UNDO / REDO
  // ════════════════════════════════════════════════════════

  function takeSnapshot() {
    return {
      annotations: JSON.parse(JSON.stringify(state.annotations)),
      selectedIdx: state.selectedIdx,
    };
  }

  function applySnapshot(snap) {
    state.annotations = JSON.parse(JSON.stringify(snap.annotations));
    state.selectedIdx = snap.selectedIdx;
    renderAnnotationList();
    render();
  }

  function pushUndo() {
    state.undoStack.push(takeSnapshot());
    if (state.undoStack.length > MAX_UNDO) state.undoStack.shift();
    state.redoStack = [];
    updateUndoButtons();
  }

  function undo() {
    if (!state.undoStack.length) return;
    state.redoStack.push(takeSnapshot());
    applySnapshot(state.undoStack.pop());
    state.dirty = true;
    updateUndoButtons();
  }

  function redo() {
    if (!state.redoStack.length) return;
    state.undoStack.push(takeSnapshot());
    applySnapshot(state.redoStack.pop());
    state.dirty = true;
    updateUndoButtons();
  }

  function updateUndoButtons() {
    document.getElementById("btnUndo").disabled = !state.undoStack.length;
    document.getElementById("btnRedo").disabled = !state.redoStack.length;
  }

  // ════════════════════════════════════════════════════════
  //  §7  SAM / MAGIC LASSO
  // ════════════════════════════════════════════════════════

  function samStatusEl() {
    let el = document.getElementById("samStatus");
    if (!el) {
      el = document.createElement("div");
      el.id = "samStatus";
      el.className = "sam-status";
      document.querySelector(".toolbar-actions").prepend(el);
    }
    return el;
  }

  async function initSAM() {
    try {
      const data = await (await fetch("/api/sam/status")).json();
      if (!data.available) {
        samStatusEl().textContent = "SAM unavailable";
        samStatusEl().title = "Install segment-anything + torch to enable";
        return;
      }
      sam.available = true;
      samStatusEl().innerHTML =
        `<span class="sam-chip">SAM ${data.model_type} · ${data.device}</span>`;
      requestSAMEmbed();
    } catch (_) { /* SAM endpoint not reachable */ }
  }

  async function requestSAMEmbed() {
    if (!sam.available) return;
    Object.assign(sam, { embedReady: false, cacheKey: null, promptPoints: [], previewPolys: null });
    samStatusEl().innerHTML += ' <span class="sam-loading">⏳ Embedding…</span>';
    try {
      const data = await (await fetch(
        `/api/project/${PID}/sam/embed/${encodeURIComponent(CURRENT)}`,
        { method: "POST" }
      )).json();
      sam.cacheKey = data.cache_key;
      data.status === "ready" ? onSAMEmbedReady() : pollSAMEmbed();
    } catch (_) {
      const el = samStatusEl().querySelector(".sam-loading");
      if (el) el.textContent = "❌ Embed failed";
    }
  }

  function pollSAMEmbed() {
    if (sam.embedPolling) clearInterval(sam.embedPolling);
    sam.embedPolling = setInterval(async () => {
      if (!sam.cacheKey) return;
      try {
        const data = await (await fetch(
          `/api/project/${PID}/sam/embed_status/${sam.cacheKey}`
        )).json();
        if (data.status === "ready") {
          clearInterval(sam.embedPolling);
          sam.embedPolling = null;
          onSAMEmbedReady();
        } else if (data.status === "error") {
          clearInterval(sam.embedPolling);
          sam.embedPolling = null;
          const el = samStatusEl().querySelector(".sam-loading");
          if (el) el.textContent = "❌ Embed failed";
        }
      } catch (_) { /* retry next tick */ }
    }, 800);
  }

  function onSAMEmbedReady() {
    sam.embedReady = true;
    const el = samStatusEl().querySelector(".sam-loading");
    if (el) el.textContent = "✅ Ready";
    setTimeout(() => { if (el) el.remove(); }, 2000);
  }

  async function samPredict() {
    if (!sam.embedReady || !sam.cacheKey || !sam.promptPoints.length || sam.busy) return;
    sam.busy = true;
    try {
      const data = await (await fetch(`/api/project/${PID}/sam/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cache_key:    sam.cacheKey,
          points:       sam.promptPoints.map(p => [p.x, p.y]),
          point_labels: sam.promptPoints.map(p => p.label),
        }),
      })).json();
      sam.previewPolys = (data.polygons && data.polygons.length) ? data.polygons : null;
      render();
    } catch (e) {
      console.error("SAM predict error:", e);
    } finally {
      sam.busy = false;
    }
  }

  function samAccept() {
    if (!sam.previewPolys || !sam.previewPolys.length) return;
    pushUndo();

    sam.previewPolys.forEach(poly => {
      state.annotations.push({
        labels:  currentLabels(),
        polygon: poly.map(([x, y]) => imageToNorm(x, y)),
      });
    });
    selectAnnotation(state.annotations.length - 1);

    state.dirty = true;
    sam.promptPoints = [];
    sam.previewPolys = null;
    renderAnnotationList();
    render();
  }

  function samCancel() {
    sam.promptPoints = [];
    sam.previewPolys = null;
    render();
  }

  // ════════════════════════════════════════════════════════
  //  §8  HIT TESTING
  // ════════════════════════════════════════════════════════

  /** Test whether image-space point (ix,iy) falls inside `ann`. */
  function hitTest(ann, ix, iy) {
    // Pure bbox (no polygon, no keypoints)
    if (ann.bbox && !ann.polygon && !ann.keypoints) {
      return hitTestBBox(ann.bbox, ix, iy);
    }
    if (ann.polygon) {
      return pointInPolygon(ix, iy, ann.polygon.map(([nx, ny]) => normToImage(nx, ny)));
    }
    if (ann.keypoints) {
      if (ann.bbox && hitTestBBox(ann.bbox, ix, iy)) return true;
      return ann.keypoints.some(k => {
        if (k.v === 0) return false;
        const [kx, ky] = normToImage(k.x, k.y);
        return Math.hypot(ix - kx, iy - ky) < KPT_HIT_R / state.zoom;
      });
    }
    return false;
  }

  function hitTestBBox(bbox, ix, iy) {
    const [cx, cy, w, h] = bbox;
    const [ax, ay] = normToImage(cx - w / 2, cy - h / 2);
    const [bx, by] = normToImage(cx + w / 2, cy + h / 2);
    return ix >= ax && ix <= bx && iy >= ay && iy <= by;
  }

  // ════════════════════════════════════════════════════════
  //  §9  ANNOTATION MUTATION HELPERS
  // ════════════════════════════════════════════════════════

  function moveAnnotation(idx, orig, dx, dy) {
    const ann = state.annotations[idx];
    const [ndx, ndy] = imageToNorm(dx, dy);
    if (orig.bbox) {
      ann.bbox = [orig.bbox[0] + ndx, orig.bbox[1] + ndy, orig.bbox[2], orig.bbox[3]];
    }
    if (orig.polygon) {
      ann.polygon = orig.polygon.map(([x, y]) => [x + ndx, y + ndy]);
    }
    if (orig.keypoints) {
      ann.keypoints = orig.keypoints.map(k => ({ x: k.x + ndx, y: k.y + ndy, v: k.v }));
    }
  }

  function recalcSkeletonBBox(ann) {
    const visible = ann.keypoints.filter(k => k.v > 0);
    if (!visible.length) return;
    const xs = visible.map(k => k.x), ys = visible.map(k => k.y);
    const pad = 0.02;
    ann.bbox = [
      (Math.min(...xs) + Math.max(...xs)) / 2,
      (Math.min(...ys) + Math.max(...ys)) / 2,
      (Math.max(...xs) - Math.min(...xs)) + pad,
      (Math.max(...ys) - Math.min(...ys)) + pad,
    ];
  }

  function deleteSelected() {
    if (state.selectedIdx < 0 || state.selectedIdx >= state.annotations.length) return;
    pushUndo();
    state.annotations.splice(state.selectedIdx, 1);
    state.selectedIdx = -1;
    state.dirty = true;
    renderAnnotationList();
    render();
  }

  function selectAnnotation(idx) {
    state.selectedIdx = idx;
    renderAnnotationList();
    render();
  }

  // ── Drawing finishers ──────────────────────────────────

  function finishPolygon() {
    const imagePts = state.drawing.points;
    pushUndo();
    const pts = imagePts.map(([x, y]) => imageToNorm(x, y));
    state.annotations.push({ labels: currentLabels(), polygon: pts });
    selectAnnotation(state.annotations.length - 1);
    state.drawing = null;
    state.dirty = true;
    render();
  }

  function finishKeypoints() {
    pushUndo();
    const kpts = state.drawing.keypoints.map(k => ({
      x: k.x / state.imgW, y: k.y / state.imgH, v: k.v,
    }));
    const xs = kpts.map(k => k.x), ys = kpts.map(k => k.y);
    const pad = 0.02;
    const bbox = [
      (Math.min(...xs) + Math.max(...xs)) / 2,
      (Math.min(...ys) + Math.max(...ys)) / 2,
      (Math.max(...xs) - Math.min(...xs)) + pad,
      (Math.max(...ys) - Math.min(...ys)) + pad,
    ];
    state.annotations.push({ labels: currentLabels(), bbox, keypoints: kpts });
    selectAnnotation(state.annotations.length - 1);
    state.drawing = null;
    state.dirty = true;
    render();
  }

  // ════════════════════════════════════════════════════════
  //  §10  MOUSE HANDLERS
  // ════════════════════════════════════════════════════════

  function onWheel(e) {
    e.preventDefault();
    const [mx, my] = getMousePos(e);
    const oldZoom = state.zoom;
    state.zoom *= e.deltaY < 0 ? 1.1 : 0.9;
    state.zoom = Math.max(0.05, Math.min(state.zoom, 20));
    state.panX = mx - (mx - state.panX) * (state.zoom / oldZoom);
    state.panY = my - (my - state.panY) * (state.zoom / oldZoom);
    render();
  }

  function onContextMenu(e) {
    e.preventDefault();
    if (state.tool === "polygon" && state.drawing &&
        state.drawing.mode === "polygon" && state.drawing.points.length >= 3) {
      finishPolygon();
    }
    if (state.tool === "magic_lasso" && sam.embedReady) {
      const [ix, iy] = screenToImage(...getMousePos(e));
      sam.promptPoints.push({ x: ix, y: iy, label: 0 });
      render();
      samPredict();
    }
  }

  function onMouseDown(e) {
    const [sx, sy] = getMousePos(e);

    // Middle mouse → always pan
    if (e.button === 1) {
      state.isPanning = true;
      state.panStart = [sx, sy];
      canvas.style.cursor = "grabbing";
      return;
    }
    if (e.button !== 0) return;

    const [ix, iy] = screenToImage(sx, sy);

    // ── Select tool ────────────────────────
    if (state.tool === "select") {
      // 1. Vertex grab on selected polygon
      const selAnn = state.annotations[state.selectedIdx];
      if (selAnn) {
        const vIdx = tryGrabVertex(selAnn, sx, sy);
        if (vIdx !== null) {
          pushUndo();
          state.draggingVertex = vIdx;
          canvas.style.cursor = "grabbing";
          return;
        }
      }
      // 2. Hit-test annotations (top-most first)
      const hit = findTopmostHit(ix, iy);
      if (hit >= 0) {
        selectAnnotation(hit);
        pushUndo();
        state.drawing = {
          mode: "move", idx: hit, startIx: ix, startIy: iy,
          orig: JSON.parse(JSON.stringify(state.annotations[hit])),
        };
        canvas.style.cursor = "grabbing";
        return;
      }
      // 3. Pan
      selectAnnotation(-1);
      state.isPanning = true;
      state.panStart = [sx, sy];
      canvas.style.cursor = "grabbing";
      return;
    }

    // ── Bbox tool ──────────────────────────
    if (state.tool === "bbox") {
      state.drawing = { mode: "bbox", x0: ix, y0: iy, x1: ix, y1: iy };
      return;
    }

    // ── Polygon tool ───────────────────────
    if (state.tool === "polygon") {
      if (!state.drawing) {
        state.drawing = { mode: "polygon", points: [[ix, iy]] };
      } else {
        state.drawing.points.push([ix, iy]);
      }
      render();
      return;
    }

    // ── Keypoint tool ──────────────────────
    if (state.tool === "keypoint") {
      if (!state.drawing) {
        state.drawing = { mode: "keypoint", keypoints: [] };
      }
      if (state.drawing.keypoints.length < keypointNames.length) {
        state.drawing.keypoints.push({ x: ix, y: iy, v: 2 });
        if (state.drawing.keypoints.length === keypointNames.length) finishKeypoints();
      }
      render();
      return;
    }

    // ── Magic Lasso tool ───────────────────
    if (state.tool === "magic_lasso") {
      if (!sam.embedReady) return;
      sam.promptPoints.push({ x: ix, y: iy, label: e.shiftKey ? 0 : 1 });
      render();
      samPredict();
      return;
    }

  }

  /** Try to grab a polygon vertex or keypoint on `ann`.
   *  Returns a draggingVertex descriptor or null. */
  function tryGrabVertex(ann, sx, sy) {
    if (ann.polygon) {
      for (let pi = 0; pi < ann.polygon.length; pi++) {
        const [vsx, vsy] = imageToScreen(...normToImage(...ann.polygon[pi]));
        if (isNearScreen(sx, sy, vsx, vsy, VERTEX_GRAB_R)) {
          return { annIdx: state.selectedIdx, ptIdx: pi };
        }
      }
    }
    if (ann.keypoints) {
      for (let ki = 0; ki < ann.keypoints.length; ki++) {
        const k = ann.keypoints[ki];
        if (k.v === 0) continue;
        const [ksx, ksy] = imageToScreen(...normToImage(k.x, k.y));
        if (isNearScreen(sx, sy, ksx, ksy, VERTEX_GRAB_R)) {
          return { annIdx: state.selectedIdx, kptIdx: ki };
        }
      }
    }
    return null;
  }

  function findTopmostHit(ix, iy) {
    for (let i = state.annotations.length - 1; i >= 0; i--) {
      if (hitTest(state.annotations[i], ix, iy)) return i;
    }
    return -1;
  }

  function onMouseMove(e) {
    const [sx, sy] = getMousePos(e);
    const [ix, iy] = screenToImage(sx, sy);

    // Coords HUD
    const info = document.getElementById("canvasInfo");
    if (ix >= 0 && ix <= state.imgW && iy >= 0 && iy <= state.imgH) {
      info.textContent = `${Math.round(ix)}, ${Math.round(iy)} | Zoom: ${(state.zoom * 100).toFixed(0)}%`;
    }

    // Panning
    if (state.isPanning && state.panStart) {
      state.panX += sx - state.panStart[0];
      state.panY += sy - state.panStart[1];
      state.panStart = [sx, sy];
      render();
      return;
    }

    // Vertex dragging
    if (state.draggingVertex) {
      const dv = state.draggingVertex;
      const ann = state.annotations[dv.annIdx];
      if (dv.ptIdx !== undefined && ann.polygon) {
        ann.polygon[dv.ptIdx] = imageToNorm(ix, iy);
      } else if (dv.kptIdx !== undefined && ann.keypoints) {
        ann.keypoints[dv.kptIdx].x = ix / state.imgW;
        ann.keypoints[dv.kptIdx].y = iy / state.imgH;
      }
      state.dirty = true;
      render();
      return;
    }

    // Active drawing
    if (state.drawing) {
      handleDrawingMove(sx, sy, ix, iy, e);
      return;
    }

    // Hover cursor for select tool
    if (state.tool === "select") {
      updateSelectCursor(sx, sy, ix, iy);
    }
  }

  function handleDrawingMove(sx, sy, ix, iy, e) {
    const d = state.drawing;
    if (d.mode === "bbox") {
      d.x1 = ix; d.y1 = iy;
      render();
    } else if (d.mode === "move") {
      moveAnnotation(d.idx, d.orig, ix - d.startIx, iy - d.startIy);
      render();
    } else if (d.mode === "polygon" && d.points.length > 0) {
      render();
      const last = d.points[d.points.length - 1];
      const [lx, ly] = imageToScreen(last[0], last[1]);
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(sx, sy);
      ctx.strokeStyle = "rgba(74,108,247,0.5)";
      ctx.lineWidth = 1; ctx.stroke();
    } else if (d.mode === "keypoint") {
      render();
    }
  }

  function updateSelectCursor(sx, sy, ix, iy) {
    const selAnn = state.annotations[state.selectedIdx];
    if (selAnn) {
      // Check handles
      if (selAnn.polygon) {
        for (const pt of selAnn.polygon) {
          const [vsx, vsy] = imageToScreen(...normToImage(...pt));
          if (isNearScreen(sx, sy, vsx, vsy, VERTEX_GRAB_R)) {
            canvas.style.cursor = "pointer"; return;
          }
        }
      }
      if (selAnn.keypoints) {
        for (const k of selAnn.keypoints) {
          if (k.v === 0) continue;
          const [ksx, ksy] = imageToScreen(...normToImage(k.x, k.y));
          if (isNearScreen(sx, sy, ksx, ksy, VERTEX_GRAB_R)) {
            canvas.style.cursor = "pointer"; return;
          }
        }
      }
    }
    canvas.style.cursor = findTopmostHit(ix, iy) >= 0 ? "pointer" : "grab";
  }

  function onMouseUp(e) {
    // End pan
    if (e.button === 1 || (state.isPanning && e.button === 0)) {
      state.isPanning = false;
      state.panStart = null;
      if (state.tool === "select") canvas.style.cursor = "grab";
      if (e.button === 1 || !state.drawing) return;
    }

    // End vertex drag
    if (state.draggingVertex) {
      const ann = state.annotations[state.draggingVertex.annIdx];
      if (ann.keypoints && ann.bbox) recalcSkeletonBBox(ann);
      state.draggingVertex = null;
      state.dirty = true;
      if (state.tool === "select") canvas.style.cursor = "grab";
      renderAnnotationList();
      return;
    }

    if (!state.drawing) return;

    const d = state.drawing;

    if (d.mode === "bbox") {
      const x0 = Math.min(d.x0, d.x1), y0 = Math.min(d.y0, d.y1);
      const x1 = Math.max(d.x0, d.x1), y1 = Math.max(d.y0, d.y1);
      if (x1 - x0 > 3 && y1 - y0 > 3) {
        pushUndo();
        const [ncx, ncy] = imageToNorm((x0 + x1) / 2, (y0 + y1) / 2);
        const [nw, nh]   = imageToNorm(x1 - x0, y1 - y0);
        state.annotations.push({ labels: currentLabels(), bbox: [ncx, ncy, nw, nh] });
        selectAnnotation(state.annotations.length - 1);
        state.dirty = true;
      }
      state.drawing = null;
      render();
    } else if (d.mode === "move") {
      state.dirty = true;
      state.drawing = null;
      if (state.tool === "select") canvas.style.cursor = "grab";
    }
  }

  function onDblClick(e) {
    if (state.tool === "polygon" && state.drawing &&
        state.drawing.mode === "polygon" && state.drawing.points.length >= 3) {
      finishPolygon();
      return;
    }
    if (state.tool !== "select") return;

    const selAnn = state.annotations[state.selectedIdx];
    if (!selAnn || !selAnn.polygon) return;

    const [sx, sy] = getMousePos(e);

    // Double-click vertex → delete it (if >3 vertices remain)
    for (let pi = 0; pi < selAnn.polygon.length; pi++) {
      const [vsx, vsy] = imageToScreen(...normToImage(...selAnn.polygon[pi]));
      if (isNearScreen(sx, sy, vsx, vsy, VERTEX_GRAB_R)) {
        if (selAnn.polygon.length > 3) {
          pushUndo();
          selAnn.polygon.splice(pi, 1);
          state.dirty = true;
          render();
        }
        return;
      }
    }
    // Double-click edge → insert vertex
    for (let pi = 0; pi < selAnn.polygon.length; pi++) {
      const ni = (pi + 1) % selAnn.polygon.length;
      const [asx, asy] = imageToScreen(...normToImage(...selAnn.polygon[pi]));
      const [bsx, bsy] = imageToScreen(...normToImage(...selAnn.polygon[ni]));
      if (distPointToSegment(sx, sy, asx, asy, bsx, bsy) < 6) {
        pushUndo();
        selAnn.polygon.splice(ni, 0, imageToNorm(...screenToImage(sx, sy)));
        state.dirty = true;
        render();
        return;
      }
    }
  }

  // ════════════════════════════════════════════════════════
  //  §11  RENDERING
  // ════════════════════════════════════════════════════════

  function render() {
    const z = state.zoom;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(z, z);

    ctx.drawImage(state.img, 0, 0);

    // Committed annotations
    state.annotations.forEach((ann, i) => drawAnnotation(ann, i === state.selectedIdx));

    // In-progress drawing
    if (state.drawing) renderDrawingPreview(z);

    // SAM prompt points
    if (sam.promptPoints.length) renderSAMPrompts(z);

    ctx.restore();

    // HUD (screen-space)
    renderHUD(z);
  }

  function renderDrawingPreview(z) {
    const d = state.drawing;
    if (d.mode === "bbox") {
      const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
      const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
      ctx.strokeStyle = DRAWING_COLOR;
      ctx.lineWidth = 2 / z;
      ctx.setLineDash([6 / z, 3 / z]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    } else if (d.mode === "polygon" && d.points.length > 0) {
      ctx.beginPath();
      d.points.forEach(([x, y], j) => j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.strokeStyle = DRAWING_COLOR;
      ctx.lineWidth = 2 / z;
      ctx.stroke();
      d.points.forEach(([x, y], j) => {
        ctx.beginPath();
        ctx.arc(x, y, 4 / z, 0, Math.PI * 2);
        ctx.fillStyle = j === 0 ? "#ff4444" : DRAWING_COLOR;
        ctx.fill();
      });
    } else if (d.mode === "keypoint") {
      // Skeleton edges between placed keypoints
      skeletonEdges.forEach(([a, b]) => {
        if (a < d.keypoints.length && b < d.keypoints.length) {
          const ka = d.keypoints[a], kb = d.keypoints[b];
          ctx.beginPath(); ctx.moveTo(ka.x, ka.y); ctx.lineTo(kb.x, kb.y);
          ctx.strokeStyle = "rgba(255,255,255,0.7)";
          ctx.lineWidth = 2 / z; ctx.stroke();
        }
      });
      d.keypoints.forEach((k, ki) => {
        ctx.beginPath();
        ctx.arc(k.x, k.y, 5 / z, 0, Math.PI * 2);
        ctx.fillStyle = BASE_PALETTE[ki % BASE_PALETTE.length];
        ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1 / z; ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.font = `${10 / z}px sans-serif`;
        ctx.fillText(keypointNames[ki] || ki, k.x + 6 / z, k.y - 6 / z);
      });
    }
  }

  function renderSAMPrompts(z) {
    sam.promptPoints.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6 / z, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#22cc44" : "#ee3333";
      ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 / z; ctx.stroke();
      // "+" or "−" label inside
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${10 / z}px sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(p.label === 1 ? "+" : "−", p.x, p.y);
      ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
    });
  }

  function renderHUD() {
    if (state.drawing) {
      if (state.drawing.mode === "keypoint") {
        const nextIdx = state.drawing.keypoints.length;
        if (nextIdx < keypointNames.length) {
          ctx.fillStyle = DRAWING_COLOR; ctx.font = "14px sans-serif";
          ctx.fillText(`Click to place: ${keypointNames[nextIdx]} (${nextIdx + 1}/${keypointNames.length})`,
                       10, canvas.height - 10);
        }
      }
      if (state.drawing.mode === "polygon") {
        ctx.fillStyle = DRAWING_COLOR; ctx.font = "13px sans-serif";
        ctx.fillText(`Polygon: ${state.drawing.points.length} pts — right-click or dbl-click to close (min 3)`,
                     10, canvas.height - 10);
      }
    }
    if (state.tool === "magic_lasso") {
      ctx.font = "13px sans-serif";
      if (!sam.embedReady) {
        ctx.fillStyle = "#f59e0b";
        ctx.fillText("⏳ SAM embedding loading…", 10, canvas.height - 10);
      } else if (sam.busy) {
        ctx.fillStyle = "#f59e0b";
        ctx.fillText("⏳ Predicting…", 10, canvas.height - 10);
      } else if (!sam.promptPoints.length) {
        ctx.fillStyle = "#888";
        const action = "Enter = accept";
        ctx.fillText(`Left-click = foreground · Right-click/Shift+click = background · ${action} · Esc = cancel`,
                     10, canvas.height - 10);
      } else {
        ctx.fillStyle = "#4a6cf7";
        const action = "Enter to accept";
        ctx.fillText(`${sam.promptPoints.length} prompt point(s) — ${action} · Esc to cancel`,
                     10, canvas.height - 10);
      }
    }
    if (task === "semantic_segmentation") {
      ctx.fillStyle = "#4a6cf7";
      ctx.font = "12px sans-serif";
      ctx.fillText("Semantic mode: overlap is auto-trimmed on save; full image coverage is still required.",
                   10, 18);
    }
  }

  function drawAnnotation(ann, selected) {
    const color = colorForLabels(ann.labels);
    const z = state.zoom;
    const lw = (selected ? 3 : 2) / z;
    const labelText = Object.values(ann.labels).join(", ");

    // BBox (pure — no keypoints overlay)
    if (ann.bbox && !ann.keypoints) {
      const [x0, y0, x1, y1] = bboxToCorners(ann.bbox);
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = color;
      ctx.font = `bold ${12 / z}px sans-serif`;
      ctx.fillText(labelText, x0, y0 - 3 / z);
      ctx.fillStyle = color + "22";
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }

    // Polygon
    if (ann.polygon) {
      const pts = ann.polygon.map(([nx, ny]) => normToImage(nx, ny));
      ctx.beginPath();
      pts.forEach(([x, y], j) => j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.closePath();
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
      ctx.fillStyle = task === "semantic_segmentation" ? (color + SEMANTIC_FILL_ALPHA) : (color + "33");
      ctx.fill();
      pts.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, (selected ? 4 : 2.5) / z, 0, Math.PI * 2);
        ctx.fillStyle = selected ? color : color + "AA";
        ctx.fill();
        if (selected) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1 / z; ctx.stroke(); }
      });
      ctx.fillStyle = color;
      ctx.font = `bold ${12 / z}px sans-serif`;
      ctx.fillText(labelText, pts[0][0], pts[0][1] - 3 / z);
    }

    // Keypoints / Skeleton
    if (ann.keypoints) {
      drawSkeleton(ann, color, selected, z);
      ctx.fillStyle = color;
      ctx.font = `bold ${12 / z}px sans-serif`;
      const [lx, ly] = ann.bbox
        ? normToImage(ann.bbox[0] - ann.bbox[2] / 2, ann.bbox[1] - ann.bbox[3] / 2)
        : [0, 0];
      ctx.fillText(labelText, lx, ly - 3 / z);
    }
  }

  /** Convert normalized [cx, cy, w, h] → image-space [x0, y0, x1, y1]. */
  function bboxToCorners(bbox) {
    const [cx, cy, w, h] = bbox;
    const [x0, y0] = normToImage(cx - w / 2, cy - h / 2);
    const [x1, y1] = normToImage(cx + w / 2, cy + h / 2);
    return [x0, y0, x1, y1];
  }

  function drawSkeleton(ann, color, selected, z) {
    // Edges
    skeletonEdges.forEach(([a, b]) => {
      if (a < ann.keypoints.length && b < ann.keypoints.length) {
        const ka = ann.keypoints[a], kb = ann.keypoints[b];
        if (ka.v > 0 && kb.v > 0) {
          const [ax, ay] = normToImage(ka.x, ka.y);
          const [bx, by] = normToImage(kb.x, kb.y);
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
          ctx.strokeStyle = color; ctx.lineWidth = 2 / z; ctx.stroke();
        }
      }
    });
    // Points
    ann.keypoints.forEach((k, ki) => {
      if (k.v === 0) return;
      const [kx, ky] = normToImage(k.x, k.y);
      ctx.beginPath();
      ctx.arc(kx, ky, (selected ? 6 : 5) / z, 0, Math.PI * 2);
      ctx.fillStyle = k.v === 1 ? "#999" : BASE_PALETTE[ki % BASE_PALETTE.length];
      ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 / z; ctx.stroke();
      if (selected) {
        ctx.fillStyle = "#fff"; ctx.font = `${9 / z}px sans-serif`;
        ctx.fillText(keypointNames[ki] || ki, kx + 7 / z, ky - 5 / z);
      }
    });
    // Dashed bbox
    if (ann.bbox) {
      const [x0, y0, x1, y1] = bboxToCorners(ann.bbox);
      ctx.strokeStyle = color; ctx.lineWidth = 1 / z;
      ctx.setLineDash([4 / z, 4 / z]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);
    }
  }

  // ════════════════════════════════════════════════════════
  //  §12  ANNOTATION LIST UI PANEL
  // ════════════════════════════════════════════════════════

  /** Build a <select> for one label group, wired to onChange + sticky save. */
  function makeLabelSelect(group, currentValue, onChange) {
    const sel = document.createElement("select");
    group.labels.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l.name; opt.textContent = l.name;
      if (l.description) opt.title = l.description;
      if (currentValue === l.name) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      onChange(sel.value);
      if (!state.stickyLabels) state.stickyLabels = {};
      state.stickyLabels[group.name] = sel.value;
      saveStickyLabels(state.stickyLabels);
    });
    return sel;
  }

  function renderAnnotationList() {
    const list   = document.getElementById("annList");
    const editor = document.getElementById("labelEditor");
    const fields = document.getElementById("labelFields");

    if (task === "classification") {
      renderClassificationPanel(list, editor, fields);
      return;
    }
    renderSpatialPanel(list, editor, fields);
  }

  function renderClassificationPanel(list, editor, fields) {
    list.innerHTML = ""; fields.innerHTML = "";
    editor.style.display = "";
    if (!state.annotations.length) state.annotations.push({ labels: currentLabels() });
    const ann = state.annotations[0];
    labelGroups.forEach(g => {
      const lbl = document.createElement("label");
      lbl.textContent = g.name + ": ";
      lbl.appendChild(makeLabelSelect(g, ann.labels[g.name], v => {
        pushUndo(); ann.labels[g.name] = v; state.dirty = true;
      }));
      fields.appendChild(lbl);
    });
  }

  function renderSpatialPanel(list, editor, fields) {
    list.innerHTML = "";

    // Sticky label indicator
    if (state.stickyLabels && Object.keys(state.stickyLabels).length) {
      const indicator = document.createElement("div");
      indicator.className = "sticky-indicator";
      const parts = labelGroups.map(g => state.stickyLabels[g.name] || "?");
      const stickyColor = colorForLabels(state.stickyLabels);
      indicator.innerHTML =
        `<span class="sticky-dot" style="background:${stickyColor}"></span>` +
        `<span class="muted" style="font-size:11px">Next: ${parts.join(", ")}</span>`;
      list.appendChild(indicator);
    }

    // Annotation items
    state.annotations.forEach((ann, i) => {
      const div = document.createElement("div");
      div.className = "ann-item" + (i === state.selectedIdx ? " selected" : "");
      const labelStr = Object.values(ann.labels).join(" | ");
      const annColor = colorForLabels(ann.labels);
      div.innerHTML =
        `<span><span class="ann-color-dot" style="background:${annColor}"></span>` +
        `#${i + 1} ${labelStr}</span>` +
        `<span class="ann-del" data-i="${i}">×</span>`;
      div.addEventListener("click", e => {
        if (e.target.classList.contains("ann-del")) {
          pushUndo();
          state.annotations.splice(parseInt(e.target.dataset.i), 1);
          state.selectedIdx = -1;
          state.dirty = true;
          renderAnnotationList();
          render();
        } else {
          selectAnnotation(i);
        }
      });
      list.appendChild(div);
    });

    // Selected annotation editor
    if (state.selectedIdx >= 0 && state.selectedIdx < state.annotations.length) {
      editor.style.display = ""; fields.innerHTML = "";
      const ann = state.annotations[state.selectedIdx];
      labelGroups.forEach(g => {
        const lbl = document.createElement("label");
        lbl.textContent = g.name + ": ";
        lbl.appendChild(makeLabelSelect(g, ann.labels[g.name], v => {
          pushUndo(); ann.labels[g.name] = v;
          state.dirty = true; renderAnnotationList(); render();
        }));
        fields.appendChild(lbl);
      });
      if (ann.polygon) {
        const p = document.createElement("p");
        p.className = "muted"; p.style.fontSize = "11px"; p.style.marginTop = "8px";
        p.textContent = `${ann.polygon.length} vertices — drag to move, dbl-click vertex to delete, dbl-click edge to add`;
        fields.appendChild(p);
      }
      if (ann.keypoints) {
        fields.appendChild(buildKeypointEditor(ann));
      }
    } else {
      editor.style.display = "none";
    }
  }

  function buildKeypointEditor(ann) {
    const kDiv = document.createElement("div");
    kDiv.innerHTML = "<h4 style='margin-top:8px'>Keypoints</h4>";
    ann.keypoints.forEach((k, ki) => {
      const row = document.createElement("label");
      row.style.fontSize = "12px";
      const sel = document.createElement("select");
      sel.style.width = "auto"; sel.style.marginLeft = "4px";
      [["2","visible"],["1","occluded"],["0","not labeled"]].forEach(([v, t]) => {
        const opt = document.createElement("option");
        opt.value = v; opt.textContent = t;
        if (k.v === parseInt(v)) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", () => {
        pushUndo(); k.v = parseInt(sel.value); state.dirty = true; render();
      });
      row.textContent = (keypointNames[ki] || `kpt${ki}`) + " ";
      row.appendChild(sel);
      kDiv.appendChild(row);
    });
    const info = document.createElement("p");
    info.className = "muted"; info.style.fontSize = "11px";
    info.textContent = "Drag keypoints to reposition";
    kDiv.appendChild(info);
    return kDiv;
  }

  // ════════════════════════════════════════════════════════
  //  §13  LOAD / SAVE (API)
  // ════════════════════════════════════════════════════════

  async function loadAnnotation(filename) {
    const data = await (await fetch(
      `/api/project/${PID}/annotation/${encodeURIComponent(filename)}`
    )).json();
    state.annotations = data.annotations || [];
    state.selectedIdx = -1;
    state.dirty = false;
    state.undoStack = []; state.redoStack = [];
    updateUndoButtons();
    renderAnnotationList();
    render();
  }

  async function save() {
    const btn = document.getElementById("btnSave");
    const body = { annotations: state.annotations };

    const resp = await fetch(`/api/project/${PID}/annotation/${encodeURIComponent(CURRENT)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      let msg = `Save failed (${resp.status})`;
      try {
        const data = await resp.json();
        if (data && data.error) msg = data.error;
      } catch (_) { /* ignore json parse error */ }
      console.error("Save failed:", msg);
      btn.textContent = "❌ Save failed";
      setTimeout(() => { btn.textContent = "💾 Save"; }, 1800);
      alert(msg);
      return;
    }

    state.dirty = false;
    const dot = document.getElementById("dot-" + CURRENT_IDX);
    if (dot) dot.classList.add("labeled");
    btn.textContent = "✓ Saved";

    setTimeout(() => { btn.textContent = "💾 Save"; }, 1200);
  }

  // ════════════════════════════════════════════════════════
  //  §14  NAVIGATION
  // ════════════════════════════════════════════════════════

  function navigate(delta) {
    if (state.dirty && !confirm("Unsaved changes. Discard?")) return;
    const idx = Math.min(Math.max(CURRENT_IDX + delta, 0), IMAGES.length - 1);
    if (idx !== CURRENT_IDX) {
      location.href = `/project/${PID}/annotate/${encodeURIComponent(IMAGES[idx])}`;
    }
  }

  window.navigateTo = function (name) {
    if (state.dirty && !confirm("Unsaved changes. Discard?")) return;
    location.href = `/project/${PID}/annotate/${encodeURIComponent(name)}`;
  };

  // ════════════════════════════════════════════════════════
  //  §15  SETUP & INIT
  // ════════════════════════════════════════════════════════

  function setupTools() {
    const bar = document.getElementById("toolButtons");
    const toolDefs = {
      classification:        [["select","🏷️ Classify"]],
      detection:             [["select","👆 Select"],["bbox","📦 Box"]],
      instance_segmentation: [["select","👆 Select"],["polygon","💎 Polygon"],["magic_lasso","✨ Magic Lasso"]],
      semantic_segmentation: [["select","👆 Select"],["polygon","✂️ Mask"],["magic_lasso","✨ Magic Lasso"]],
      skeleton:              [["select","👆 Select"],["keypoint","🏃🏻 Skeleton"]],
    };
    const defs = toolDefs[task] || [];
    defs.forEach(([t, label]) => {
      const btn = document.createElement("button");
      btn.className = "btn-sm"; btn.textContent = label; btn.dataset.tool = t;
      btn.addEventListener("click", () => setTool(t));
      bar.appendChild(btn);
    });
    if (defs[0]) setTool(defs[0][0]);
    if (defs.some(([t]) => t === "magic_lasso")) initSAM();
  }

  function setTool(t) {
    state.tool = t;
    state.drawing = null;
    state.draggingVertex = null;
    document.querySelectorAll(".toolbar-tools .btn-sm").forEach(b => {
      b.classList.toggle("active", b.dataset.tool === t);
    });
    canvas.style.cursor = (t === "select") ? "grab" : "crosshair";
    render();
  }

  function setupKeyboard() {
    document.addEventListener("keydown", e => {
      if (["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key === "z") { e.preventDefault(); undo(); return; }
      if (mod && e.key === "y") { e.preventDefault(); redo(); return; }
      if (mod && e.key === "s") { e.preventDefault(); save(); return; }
      if (e.key === "a" || e.key === "A") navigate(-1);
      if (e.key === "d" || e.key === "D") navigate(1);
      if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
      if (e.key === "Escape") {
        if (state.tool === "magic_lasso" && sam.promptPoints.length) { samCancel(); return; }
        state.drawing = null; state.draggingVertex = null; render();
      }
      if (e.key === "Enter" && state.tool === "magic_lasso") samAccept();
    });
  }

  function setupMouse() {
    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("mousemove",  onMouseMove);
    canvas.addEventListener("mouseup",    onMouseUp);
    canvas.addEventListener("dblclick",   onDblClick);
    canvas.addEventListener("wheel",      onWheel, { passive: false });
    canvas.addEventListener("contextmenu", onContextMenu);
  }

  function loadImage(filename) {
    state.img = new Image();
    state.img.onload = () => {
      state.imgW = state.img.naturalWidth;
      state.imgH = state.img.naturalHeight;
      fitCanvas();
      loadAnnotation(filename);
    };
    state.img.src = `/api/project/${PID}/image/${encodeURIComponent(filename)}`;
  }

  function fitCanvas() {
    const rect = wrap.getBoundingClientRect();
    canvas.width  = rect.width;
    canvas.height = rect.height;
    state.zoom = Math.min(rect.width / state.imgW, rect.height / state.imgH) * 0.95;
    state.panX = (rect.width  - state.imgW * state.zoom) / 2;
    state.panY = (rect.height - state.imgH * state.zoom) / 2;
    render();
  }

  /** Prefetch annotation dots for sidebar thumbnails. */
  function prefetchAnnotationDots() {
    fetch(`/api/project/${PID}/labeled_status`)
      .then(r => r.json())
      .then(statusMap => {
        IMAGES.forEach((name, i) => {
          if (statusMap[name]) {
            const dot = document.getElementById("dot-" + i);
            if (dot) dot.classList.add("labeled");
          }
        });
      });
  }

  // ── Boot ───────────────────────────────────────────────
  function init() {
    loadStickyLabels();
    setupTools();
    setupKeyboard();
    setupMouse();
    loadImage(CURRENT);


    prefetchAnnotationDots();

    window.addEventListener("resize", fitCanvas);
    document.getElementById("btnPrev").addEventListener("click", () => navigate(-1));
    document.getElementById("btnNext").addEventListener("click", () => navigate(1));
    document.getElementById("btnSave").addEventListener("click", save);
    document.getElementById("btnUndo").addEventListener("click", undo);
    document.getElementById("btnRedo").addEventListener("click", redo);
  }

  init();
})();
