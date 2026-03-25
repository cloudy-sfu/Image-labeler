/*
  Canvas-based annotation engine for Flask Image Labeler.
  Supports bbox, polygon (with vertex editing), mask painting,
  keypoint/skeleton, classification.
  Undo/redo, right-click to close polygon.
  Ref: YOLO annotation format — https://docs.ultralytics.com/datasets/detect/
*/

(function () {
  "use strict";

  const canvas = document.getElementById("mainCanvas");
  const ctx = canvas.getContext("2d");
  const wrap = canvas.parentElement;

  // ── State ──────────────────────────────────────────────
  let img = new Image();
  let imgW = 1, imgH = 1;
  let zoom = 1, panX = 0, panY = 0;
  let annotations = [];
  let selectedIdx = -1;
  let tool = "";
  let drawing = null;
  let dirty = false;
  let isPanning = false, panStart = null;

  // Vertex editing
  let draggingVertex = null;

  // Mask (semantic seg)
  let maskCanvas = null, maskCtx = null;
  let lastBrushPos = null;

  // Undo / Redo stacks
  let undoStack = [];
  let redoStack = [];
  const MAX_UNDO = 60;

  // ── Sticky (remembered) labels ─────────────────────────
  // Persisted per-project in sessionStorage so it survives image navigation
  const STICKY_KEY = "labeler_sticky_" + PID;
  let stickyLabels = null; // { groupName: labelName, ... }

  function loadStickyLabels() {
    try {
      const raw = sessionStorage.getItem(STICKY_KEY);
      if (raw) { stickyLabels = JSON.parse(raw); return; }
    } catch (_) {}
    stickyLabels = null;
  }

  function saveStickyLabels(labels) {
    stickyLabels = Object.assign({}, labels);
    try { sessionStorage.setItem(STICKY_KEY, JSON.stringify(stickyLabels)); } catch (_) {}
  }

  // ── Deterministic color from label combination ─────────
  const BASE_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324","#800000","#aaffc3","#808000","#000075",
    "#f44336","#9c27b0","#2196f3","#009688","#ff9800","#795548",
    "#607d8b","#e91e63","#00bcd4","#8bc34a","#ffc107","#673ab7"
  ];
  const colorCache = {};

  function colorForLabels(labels) {
    // Build a stable string key from the labels dict
    const key = labelGroups.map(g => labels[g.name] || "").join("\x00");
    if (colorCache[key]) return colorCache[key];
    // Deterministic hash → palette index
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
    }
    const idx = Math.abs(hash) % BASE_PALETTE.length;
    colorCache[key] = BASE_PALETTE[idx];
    return colorCache[key];
  }

  // Fallback color for in-progress drawing (before labels are assigned)
  const DRAWING_COLOR = "#4a6cf7";

  const task = PROJECT.task;
  const labelGroups = PROJECT.label_groups || [];
  const keypointNames = PROJECT.keypoint_names || [];
  const skeletonEdges = PROJECT.skeleton_edges || [];

  // ── Undo / Redo ────────────────────────────────────────
  function pushUndo() {
    const snap = takeSnapshot();
    undoStack.push(snap);
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    redoStack = [];
    updateUndoButtons();
  }

  function takeSnapshot() {
    const snap = {
      annotations: JSON.parse(JSON.stringify(annotations)),
      selectedIdx: selectedIdx,
    };
    if (task === "semantic_segmentation" && maskCanvas) {
      snap.maskData = maskCanvas.toDataURL("image/png");
    }
    return snap;
  }

  function applySnapshot(snap) {
    annotations = JSON.parse(JSON.stringify(snap.annotations));
    selectedIdx = snap.selectedIdx;
    if (task === "semantic_segmentation" && snap.maskData && maskCtx) {
      const mi = new Image();
      mi.onload = () => {
        maskCtx.clearRect(0, 0, imgW, imgH);
        maskCtx.drawImage(mi, 0, 0);
        renderAnnotationList();
        render();
      };
      mi.src = snap.maskData;
      return;
    }
    renderAnnotationList();
    render();
  }

  function undo() {
    if (!undoStack.length) return;
    redoStack.push(takeSnapshot());
    applySnapshot(undoStack.pop());
    dirty = true;
    updateUndoButtons();
  }

  function redo() {
    if (!redoStack.length) return;
    undoStack.push(takeSnapshot());
    applySnapshot(redoStack.pop());
    dirty = true;
    updateUndoButtons();
  }

  function updateUndoButtons() {
    document.getElementById("btnUndo").disabled = !undoStack.length;
    document.getElementById("btnRedo").disabled = !redoStack.length;
  }

  // ── Init ───────────────────────────────────────────────
  function init() {
    loadStickyLabels();
    setupTools();
    setupNavigation();
    setupKeyboard();
    setupMouse();
    loadImage(CURRENT);

    if (task === "semantic_segmentation") {
      document.getElementById("brushSizeWrap").style.display = "";
      document.getElementById("brushSize").addEventListener("input", e => {
        document.getElementById("brushSizeVal").textContent = e.target.value;
      });
    }

    IMAGES.forEach((name, i) => {
      fetch(`/api/project/${PID}/annotation/${encodeURIComponent(name)}`)
        .then(r => r.json()).then(d => {
          const has = (d.annotations && d.annotations.length) || d.mask;
          if (has) {
            const dot = document.getElementById("dot-" + i);
            if (dot) dot.classList.add("labeled");
          }
        });
    });

    window.addEventListener("resize", fitCanvas);
    document.getElementById("btnUndo").addEventListener("click", undo);
    document.getElementById("btnRedo").addEventListener("click", redo);
  }

  // ── Tool buttons ───────────────────────────────────────
  function setupTools() {
    const bar = document.getElementById("toolButtons");
    const toolDefs = {
      classification: [["select", "🏷️ Classify"]],
      detection: [["select", "👆 Select"], ["bbox", "⬜ BBox"]],
      instance_segmentation: [["select", "👆 Select"], ["polygon", "🔷 Polygon"]],
      semantic_segmentation: [["brush", "🖌️ Brush"], ["eraser", "🧹 Eraser"]],
      skeleton: [["select", "👆 Select"], ["keypoint", "🦴 Pose"]],
    };
    (toolDefs[task] || []).forEach(([t, label]) => {
      const btn = document.createElement("button");
      btn.className = "btn-sm";
      btn.textContent = label;
      btn.dataset.tool = t;
      btn.addEventListener("click", () => setTool(t));
      bar.appendChild(btn);
    });
    const dt = (toolDefs[task] || [])[0];
    if (dt) setTool(dt[0]);
  }

  function setTool(t) {
    tool = t;
    drawing = null;
    draggingVertex = null;
    document.querySelectorAll(".toolbar-tools .btn-sm").forEach(b => {
      b.classList.toggle("active", b.dataset.tool === t);
    });
    // Select tool uses default cursor; drawing tools use crosshair
    canvas.style.cursor = (t === "select") ? "grab" : "crosshair";
    render();
  }

  // ── Navigation ─────────────────────────────────────────
  function setupNavigation() {
    document.getElementById("btnPrev").addEventListener("click", () => navigate(-1));
    document.getElementById("btnNext").addEventListener("click", () => navigate(1));
    document.getElementById("btnSave").addEventListener("click", save);
  }

  function navigate(delta) {
    if (dirty && !confirm("Unsaved changes. Discard?")) return;
    const idx = Math.min(Math.max(CURRENT_IDX + delta, 0), IMAGES.length - 1);
    if (idx !== CURRENT_IDX) {
      location.href = `/project/${PID}/annotate/${encodeURIComponent(IMAGES[idx])}`;
    }
  }

  window.navigateTo = function (name) {
    if (dirty && !confirm("Unsaved changes. Discard?")) return;
    location.href = `/project/${PID}/annotate/${encodeURIComponent(name)}`;
  };

  // ── Keyboard ───────────────────────────────────────────
  function setupKeyboard() {
    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
      if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); undo(); return; }
      if ((e.ctrlKey || e.metaKey) && e.key === "y") { e.preventDefault(); redo(); return; }
      if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); return; }
      if (e.key === "a" || e.key === "A") navigate(-1);
      if (e.key === "d" || e.key === "D") navigate(1);
      if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
      if (e.key === "Escape") { drawing = null; draggingVertex = null; render(); }
    });
  }

  // ── Image loading ──────────────────────────────────────
  function loadImage(filename) {
    img = new Image();
    img.onload = () => {
      imgW = img.naturalWidth;
      imgH = img.naturalHeight;
      if (task === "semantic_segmentation") {
        maskCanvas = document.createElement("canvas");
        maskCanvas.width = imgW;
        maskCanvas.height = imgH;
        maskCtx = maskCanvas.getContext("2d");
        maskCtx.clearRect(0, 0, imgW, imgH);
      }
      fitCanvas();
      loadAnnotation(filename);
    };
    img.src = `/api/project/${PID}/image/${encodeURIComponent(filename)}`;
  }

  function fitCanvas() {
    const rect = wrap.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    zoom = Math.min(rect.width / imgW, rect.height / imgH) * 0.95;
    panX = (rect.width - imgW * zoom) / 2;
    panY = (rect.height - imgH * zoom) / 2;
    render();
  }

  // ── Coordinate transforms ──────────────────────────────
  function screenToImage(sx, sy) { return [(sx - panX) / zoom, (sy - panY) / zoom]; }
  function imageToScreen(ix, iy) { return [ix * zoom + panX, iy * zoom + panY]; }
  function normToImage(nx, ny) { return [nx * imgW, ny * imgH]; }
  function imageToNorm(ix, iy) { return [ix / imgW, iy / imgH]; }

  // ── Mouse ──────────────────────────────────────────────
  function setupMouse() {
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("dblclick", onDblClick);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("contextmenu", onContextMenu);
  }

  function getMousePos(e) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function onWheel(e) {
    e.preventDefault();
    const [mx, my] = getMousePos(e);
    const oldZoom = zoom;
    zoom *= e.deltaY < 0 ? 1.1 : 0.9;
    zoom = Math.max(0.05, Math.min(zoom, 20));
    panX = mx - (mx - panX) * (zoom / oldZoom);
    panY = my - (my - panY) * (zoom / oldZoom);
    render();
  }

  function onContextMenu(e) {
    e.preventDefault();
    if (tool === "polygon" && drawing && drawing.mode === "polygon" && drawing.points.length >= 3) {
      finishPolygon();
    }
  }

  function onMouseDown(e) {
    const [sx, sy] = getMousePos(e);

    // Middle mouse: always pan
    if (e.button === 1) {
      isPanning = true;
      panStart = [sx, sy];
      canvas.style.cursor = "grabbing";
      return;
    }
    if (e.button !== 0) return;

    const [ix, iy] = screenToImage(sx, sy);

    if (tool === "select") {
      // 1. Try vertex grab on selected polygon
      if (selectedIdx >= 0 && annotations[selectedIdx] && annotations[selectedIdx].polygon) {
        const ann = annotations[selectedIdx];
        for (let pi = 0; pi < ann.polygon.length; pi++) {
          const [vx, vy] = normToImage(ann.polygon[pi][0], ann.polygon[pi][1]);
          const [vsx, vsy] = imageToScreen(vx, vy);
          if (Math.hypot(sx - vsx, sy - vsy) < 8) {
            pushUndo();
            draggingVertex = { annIdx: selectedIdx, ptIdx: pi };
            canvas.style.cursor = "grabbing";
            return;
          }
        }
      }
      // 2. Try keypoint grab on selected skeleton
      if (selectedIdx >= 0 && annotations[selectedIdx] && annotations[selectedIdx].keypoints) {
        const ann = annotations[selectedIdx];
        for (let ki = 0; ki < ann.keypoints.length; ki++) {
          const k = ann.keypoints[ki];
          if (k.v === 0) continue;
          const [kx, ky] = normToImage(k.x, k.y);
          const [ksx, ksy] = imageToScreen(kx, ky);
          if (Math.hypot(sx - ksx, sy - ksy) < 8) {
            pushUndo();
            draggingVertex = { annIdx: selectedIdx, kptIdx: ki };
            canvas.style.cursor = "grabbing";
            return;
          }
        }
      }
      // 3. Hit test annotations
      let hit = -1;
      for (let i = annotations.length - 1; i >= 0; i--) {
        if (hitTest(annotations[i], ix, iy)) { hit = i; break; }
      }
      if (hit >= 0) {
        selectAnnotation(hit);
        pushUndo();
        drawing = { mode: "move", idx: hit, startIx: ix, startIy: iy,
                    orig: JSON.parse(JSON.stringify(annotations[hit])) };
        canvas.style.cursor = "grabbing";
        return;
      }
      // 4. Nothing hit → pan the canvas
      selectAnnotation(-1);
      isPanning = true;
      panStart = [sx, sy];
      canvas.style.cursor = "grabbing";
      return;
    }

    if (tool === "bbox") {
      drawing = { mode: "bbox", x0: ix, y0: iy, x1: ix, y1: iy };
      return;
    }

    if (tool === "polygon") {
      if (!drawing) {
        drawing = { mode: "polygon", points: [[ix, iy]] };
      } else {
        drawing.points.push([ix, iy]);
      }
      render();
      return;
    }

    if (tool === "keypoint") {
      if (!drawing) {
        drawing = { mode: "keypoint", keypoints: [] };
      }
      const kIdx = drawing.keypoints.length;
      if (kIdx < keypointNames.length) {
        drawing.keypoints.push({ x: ix, y: iy, v: 2 });
        if (drawing.keypoints.length === keypointNames.length) {
          finishKeypoints();
        }
      }
      render();
      return;
    }

    if (tool === "brush" || tool === "eraser") {
      pushUndo();
      drawing = { mode: "brush" };
      lastBrushPos = [ix, iy];
      paintMask(ix, iy);
      return;
    }
  }

  function onMouseMove(e) {
    const [sx, sy] = getMousePos(e);
    const [ix, iy] = screenToImage(sx, sy);

    const info = document.getElementById("canvasInfo");
    if (ix >= 0 && ix <= imgW && iy >= 0 && iy <= imgH) {
      info.textContent = `${Math.round(ix)}, ${Math.round(iy)} | Zoom: ${(zoom * 100).toFixed(0)}%`;
    }

    if (isPanning && panStart) {
      panX += sx - panStart[0];
      panY += sy - panStart[1];
      panStart = [sx, sy];
      render();
      return;
    }

    // Vertex dragging
    if (draggingVertex) {
      const ann = annotations[draggingVertex.annIdx];
      if (draggingVertex.ptIdx !== undefined && ann.polygon) {
        const [nx, ny] = imageToNorm(ix, iy);
        ann.polygon[draggingVertex.ptIdx] = [nx, ny];
        dirty = true;
        render();
      } else if (draggingVertex.kptIdx !== undefined && ann.keypoints) {
        ann.keypoints[draggingVertex.kptIdx].x = ix / imgW;
        ann.keypoints[draggingVertex.kptIdx].y = iy / imgH;
        dirty = true;
        render();
      }
      return;
    }

    if (drawing) {
      if (drawing.mode === "bbox") {
        drawing.x1 = ix;
        drawing.y1 = iy;
        render();
      } else if (drawing.mode === "move") {
        const dx = ix - drawing.startIx;
        const dy = iy - drawing.startIy;
        moveAnnotation(drawing.idx, drawing.orig, dx, dy);
        render();
      } else if (drawing.mode === "brush" && e.buttons === 1) {
        paintMaskInterpolated(ix, iy);
        lastBrushPos = [ix, iy];
      } else if (drawing.mode === "polygon" || drawing.mode === "keypoint") {
        render();
        if (drawing.mode === "polygon" && drawing.points.length > 0) {
          const last = drawing.points[drawing.points.length - 1];
          const [lx, ly] = imageToScreen(last[0], last[1]);
          ctx.beginPath();
          ctx.moveTo(lx, ly);
          ctx.lineTo(sx, sy);
          ctx.strokeStyle = "rgba(74,108,247,0.5)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // Update cursor for select tool hover states
    if (tool === "select" && !isPanning && !drawing && !draggingVertex) {
      let onHandle = false;
      if (selectedIdx >= 0 && annotations[selectedIdx]) {
        const ann = annotations[selectedIdx];
        if (ann.polygon) {
          for (const pt of ann.polygon) {
            const [vx, vy] = normToImage(pt[0], pt[1]);
            const [vsx, vsy] = imageToScreen(vx, vy);
            if (Math.hypot(sx - vsx, sy - vsy) < 8) { onHandle = true; break; }
          }
        }
        if (!onHandle && ann.keypoints) {
          for (const k of ann.keypoints) {
            if (k.v === 0) continue;
            const [kx, ky] = normToImage(k.x, k.y);
            const [ksx, ksy] = imageToScreen(kx, ky);
            if (Math.hypot(sx - ksx, sy - ksy) < 8) { onHandle = true; break; }
          }
        }
      }
      if (onHandle) {
        canvas.style.cursor = "pointer";
      } else {
        // Check if hovering an annotation
        let overAnn = false;
        for (let i = annotations.length - 1; i >= 0; i--) {
          if (hitTest(annotations[i], ix, iy)) { overAnn = true; break; }
        }
        canvas.style.cursor = overAnn ? "pointer" : "grab";
      }
    }
  }

  function onMouseUp(e) {
    if (e.button === 1 || (isPanning && e.button === 0)) {
      isPanning = false;
      panStart = null;
      if (tool === "select") canvas.style.cursor = "grab";
      if (e.button === 1) return;
      if (!drawing) return; // was panning, done
    }

    if (draggingVertex) {
      const ann = annotations[draggingVertex.annIdx];
      if (ann.keypoints && ann.bbox) {
        recalcSkeletonBBox(ann);
      }
      draggingVertex = null;
      dirty = true;
      if (tool === "select") canvas.style.cursor = "grab";
      renderAnnotationList();
      return;
    }

    if (!drawing) return;

    if (drawing.mode === "bbox") {
      const x0 = Math.min(drawing.x0, drawing.x1);
      const y0 = Math.min(drawing.y0, drawing.y1);
      const x1 = Math.max(drawing.x0, drawing.x1);
      const y1 = Math.max(drawing.y0, drawing.y1);
      if (x1 - x0 > 3 && y1 - y0 > 3) {
        pushUndo();
        const [ncx, ncy] = imageToNorm((x0 + x1) / 2, (y0 + y1) / 2);
        const [nw, nh] = imageToNorm(x1 - x0, y1 - y0);
        annotations.push({ labels: currentLabels(), bbox: [ncx, ncy, nw, nh] });
        selectAnnotation(annotations.length - 1);
        dirty = true;
      }
      drawing = null;
      render();
    } else if (drawing.mode === "move") {
      dirty = true;
      drawing = null;
      if (tool === "select") canvas.style.cursor = "grab";
    } else if (drawing.mode === "brush") {
      drawing = null;
      lastBrushPos = null;
      dirty = true;
    }
  }

  function onDblClick(e) {
    if (tool === "polygon" && drawing && drawing.mode === "polygon" && drawing.points.length >= 3) {
      finishPolygon();
    }
    if (tool === "select" && selectedIdx >= 0 && annotations[selectedIdx] && annotations[selectedIdx].polygon) {
      const [sx, sy] = getMousePos(e);
      const ann = annotations[selectedIdx];
      for (let pi = 0; pi < ann.polygon.length; pi++) {
        const [vx, vy] = normToImage(ann.polygon[pi][0], ann.polygon[pi][1]);
        const [vsx, vsy] = imageToScreen(vx, vy);
        if (Math.hypot(sx - vsx, sy - vsy) < 8) {
          if (ann.polygon.length > 3) {
            pushUndo();
            ann.polygon.splice(pi, 1);
            dirty = true;
            render();
          }
          return;
        }
      }
      for (let pi = 0; pi < ann.polygon.length; pi++) {
        const ni = (pi + 1) % ann.polygon.length;
        const [ax, ay] = normToImage(ann.polygon[pi][0], ann.polygon[pi][1]);
        const [bx, by] = normToImage(ann.polygon[ni][0], ann.polygon[ni][1]);
        const [asx, asy] = imageToScreen(ax, ay);
        const [bsx, bsy] = imageToScreen(bx, by);
        if (distPointToSegment(sx, sy, asx, asy, bsx, bsy) < 6) {
          pushUndo();
          const [ix, iy] = screenToImage(sx, sy);
          ann.polygon.splice(ni, 0, imageToNorm(ix, iy));
          dirty = true;
          render();
          return;
        }
      }
    }
  }

  function distPointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  // ── Current (sticky) labels ────────────────────────────
  function currentLabels() {
    // Use sticky labels if available, otherwise default to first option
    if (stickyLabels) {
      // Validate that all groups are present and values still valid
      const labels = {};
      labelGroups.forEach(g => {
        const names = g.labels.map(l => l.name);
        if (stickyLabels[g.name] && names.includes(stickyLabels[g.name])) {
          labels[g.name] = stickyLabels[g.name];
        } else {
          labels[g.name] = names[0] || "";
        }
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

  // ── Drawing finishers ──────────────────────────────────
  function finishPolygon() {
    pushUndo();
    const pts = drawing.points.map(([x, y]) => imageToNorm(x, y));
    annotations.push({ labels: currentLabels(), polygon: pts });
    selectAnnotation(annotations.length - 1);
    drawing = null;
    dirty = true;
    render();
  }

  function finishKeypoints() {
    pushUndo();
    const kpts = drawing.keypoints.map(k => ({
      x: k.x / imgW, y: k.y / imgH, v: k.v
    }));
    const xs = kpts.map(k => k.x), ys = kpts.map(k => k.y);
    const pad = 0.02;
    const bbox = [(Math.min(...xs) + Math.max(...xs)) / 2,
                  (Math.min(...ys) + Math.max(...ys)) / 2,
                  (Math.max(...xs) - Math.min(...xs)) + pad,
                  (Math.max(...ys) - Math.min(...ys)) + pad];
    annotations.push({ labels: currentLabels(), bbox, keypoints: kpts });
    selectAnnotation(annotations.length - 1);
    drawing = null;
    dirty = true;
    render();
  }

  function recalcSkeletonBBox(ann) {
    const visible = ann.keypoints.filter(k => k.v > 0);
    if (!visible.length) return;
    const xs = visible.map(k => k.x), ys = visible.map(k => k.y);
    const pad = 0.02;
    ann.bbox = [(Math.min(...xs) + Math.max(...xs)) / 2,
                (Math.min(...ys) + Math.max(...ys)) / 2,
                (Math.max(...xs) - Math.min(...xs)) + pad,
                (Math.max(...ys) - Math.min(...ys)) + pad];
  }

  function paintMask(ix, iy) {
    if (!maskCtx) return;
    const bs = parseInt(document.getElementById("brushSize").value);
    const isEraser = tool === "eraser" || document.getElementById("eraserToggle").checked;
    maskCtx.beginPath();
    maskCtx.arc(ix, iy, bs, 0, Math.PI * 2);
    if (isEraser) {
      maskCtx.globalCompositeOperation = "destination-out";
      maskCtx.fill();
      maskCtx.globalCompositeOperation = "source-over";
    } else {
      const cid = getCurrentMaskClassId();
      const r = (cid + 1) & 0xff;
      const g = ((cid + 1) >> 8) & 0xff;
      maskCtx.fillStyle = `rgba(${r},${g},0,1)`;
      maskCtx.fill();
    }
    render();
  }

  function paintMaskInterpolated(ix, iy) {
    if (!lastBrushPos) { paintMask(ix, iy); return; }
    const [lx, ly] = lastBrushPos;
    const dist = Math.hypot(ix - lx, iy - ly);
    const bs = parseInt(document.getElementById("brushSize").value);
    const steps = Math.max(1, Math.ceil(dist / (bs * 0.3)));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      paintMask(lx + (ix - lx) * t, ly + (iy - ly) * t);
    }
  }

  function getCurrentMaskClassId() {
    const labels = {};
    labelGroups.forEach(g => {
      const sel = document.getElementById("masklg-" + g.name);
      if (sel) labels[g.name] = sel.value;
    });
    return labelsToClassId(labels);
  }

  function labelsToClassId(labels) {
    if (!labelGroups.length) return 0;
    let idx = 0, multiplier = 1;
    for (let i = labelGroups.length - 1; i >= 0; i--) {
      const g = labelGroups[i];
      const names = g.labels.map(l => l.name);
      const li = names.indexOf(labels[g.name]);
      idx += (li >= 0 ? li : 0) * multiplier;
      multiplier *= g.labels.length;
    }
    return idx;
  }

  // ── Hit testing ────────────────────────────────────────
  function hitTest(ann, ix, iy) {
    if (ann.bbox && !ann.polygon && !ann.keypoints) {
      const [cx, cy, w, h] = ann.bbox;
      const [ax, ay] = normToImage(cx - w / 2, cy - h / 2);
      const [bx, by] = normToImage(cx + w / 2, cy + h / 2);
      return ix >= ax && ix <= bx && iy >= ay && iy <= by;
    }
    if (ann.polygon) {
      const pts = ann.polygon.map(([nx, ny]) => normToImage(nx, ny));
      return pointInPolygon(ix, iy, pts);
    }
    if (ann.keypoints) {
      if (ann.bbox) {
        const [cx, cy, w, h] = ann.bbox;
        const [ax, ay] = normToImage(cx - w / 2, cy - h / 2);
        const [bx, by] = normToImage(cx + w / 2, cy + h / 2);
        if (ix >= ax && ix <= bx && iy >= ay && iy <= by) return true;
      }
      for (const k of ann.keypoints) {
        if (k.v === 0) continue;
        const [kx, ky] = normToImage(k.x, k.y);
        if (Math.hypot(ix - kx, iy - ky) < 10 / zoom) return true;
      }
    }
    return false;
  }

  function pointInPolygon(x, y, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i][0], yi = pts[i][1];
      const xj = pts[j][0], yj = pts[j][1];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  // ── Moving annotations ─────────────────────────────────
  function moveAnnotation(idx, orig, dx, dy) {
    const ann = annotations[idx];
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

  // ── Annotation list panel ──────────────────────────────
  function selectAnnotation(idx) {
    selectedIdx = idx;
    renderAnnotationList();
    render();
  }

  /** Build a label selector for a group; when changed, update the annotation
   *  and persist to stickyLabels. */
  function makeLabelSelect(group, currentValue, onChange) {
    const sel = document.createElement("select");
    group.labels.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l.name;
      opt.textContent = l.name;
      if (l.description) opt.title = l.description;
      if (currentValue === l.name) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      onChange(sel.value);
      // Update sticky
      if (!stickyLabels) stickyLabels = {};
      stickyLabels[group.name] = sel.value;
      saveStickyLabels(stickyLabels);
    });
    return sel;
  }

  function renderAnnotationList() {
    const list = document.getElementById("annList");
    const editor = document.getElementById("labelEditor");
    const fields = document.getElementById("labelFields");

    if (task === "classification") {
      list.innerHTML = "";
      fields.innerHTML = "";
      editor.style.display = "";
      if (!annotations.length) annotations.push({ labels: currentLabels() });
      const ann = annotations[0];
      labelGroups.forEach(g => {
        const lbl = document.createElement("label");
        lbl.textContent = g.name + ": ";
        const sel = makeLabelSelect(g, ann.labels[g.name], v => {
          pushUndo();
          ann.labels[g.name] = v;
          dirty = true;
        });
        lbl.appendChild(sel);
        fields.appendChild(lbl);
      });
      return;
    }

    if (task === "semantic_segmentation") {
      list.innerHTML = "<p class='muted'>Paint mask on canvas. Right-click or check Eraser to erase.</p>";
      editor.style.display = "";
      fields.innerHTML = "";
      labelGroups.forEach(g => {
        const lbl = document.createElement("label");
        lbl.textContent = g.name + ": ";
        const sel = document.createElement("select");
        sel.id = "masklg-" + g.name;
        // Use sticky if available
        const stickyVal = stickyLabels && stickyLabels[g.name];
        g.labels.forEach(l => {
          const opt = document.createElement("option");
          opt.value = l.name; opt.textContent = l.name;
          if (l.description) opt.title = l.description;
          if (stickyVal ? l.name === stickyVal : false) opt.selected = true;
          sel.appendChild(opt);
        });
        sel.addEventListener("change", () => {
          if (!stickyLabels) stickyLabels = {};
          stickyLabels[g.name] = sel.value;
          saveStickyLabels(stickyLabels);
        });
        lbl.appendChild(sel);
        fields.appendChild(lbl);
      });
      return;
    }

    // Spatial annotations list
    list.innerHTML = "";

    // Show current sticky label indicator
    if (stickyLabels && Object.keys(stickyLabels).length) {
      const indicator = document.createElement("div");
      indicator.className = "sticky-indicator";
      const parts = labelGroups.map(g => stickyLabels[g.name] || "?");
      const stickyColor = colorForLabels(stickyLabels);
      indicator.innerHTML = `<span class="sticky-dot" style="background:${stickyColor}"></span>
        <span class="muted" style="font-size:11px">Next: ${parts.join(", ")}</span>`;
      list.appendChild(indicator);
    }

    annotations.forEach((ann, i) => {
      const div = document.createElement("div");
      div.className = "ann-item" + (i === selectedIdx ? " selected" : "");
      const typeStr = ann.bbox && ann.keypoints ? "🦴" : ann.bbox ? "⬜" : ann.polygon ? "🔷" : "?";
      const labelStr = Object.values(ann.labels).join(", ");
      const annColor = colorForLabels(ann.labels);
      div.innerHTML = `<span><span class="ann-color-dot" style="background:${annColor}"></span>
        ${typeStr} #${i + 1} ${labelStr}</span><span class="ann-del" data-i="${i}">×</span>`;
      div.addEventListener("click", (e) => {
        if (e.target.classList.contains("ann-del")) {
          pushUndo();
          annotations.splice(parseInt(e.target.dataset.i), 1);
          selectedIdx = -1;
          dirty = true;
          renderAnnotationList();
          render();
        } else {
          selectAnnotation(i);
        }
      });
      list.appendChild(div);
    });

    if (selectedIdx >= 0 && selectedIdx < annotations.length) {
      editor.style.display = "";
      fields.innerHTML = "";
      const ann = annotations[selectedIdx];
      labelGroups.forEach(g => {
        const lbl = document.createElement("label");
        lbl.textContent = g.name + ": ";
        const sel = makeLabelSelect(g, ann.labels[g.name], v => {
          pushUndo();
          ann.labels[g.name] = v;
          dirty = true;
          renderAnnotationList();
          render();
        });
        lbl.appendChild(sel);
        fields.appendChild(lbl);
      });

      if (ann.polygon) {
        const pInfo = document.createElement("p");
        pInfo.className = "muted";
        pInfo.style.fontSize = "11px";
        pInfo.style.marginTop = "8px";
        pInfo.textContent = `${ann.polygon.length} vertices — drag to move, dbl-click vertex to delete, dbl-click edge to add`;
        fields.appendChild(pInfo);
      }

      if (ann.keypoints) {
        const kDiv = document.createElement("div");
        kDiv.innerHTML = "<h4 style='margin-top:8px'>Keypoints</h4>";
        ann.keypoints.forEach((k, ki) => {
          const row = document.createElement("label");
          row.style.fontSize = "12px";
          const sel = document.createElement("select");
          sel.style.width = "auto";
          sel.style.marginLeft = "4px";
          [["2", "visible"], ["1", "occluded"], ["0", "not labeled"]].forEach(([v, t]) => {
            const opt = document.createElement("option");
            opt.value = v; opt.textContent = t;
            if (k.v === parseInt(v)) opt.selected = true;
            sel.appendChild(opt);
          });
          sel.addEventListener("change", () => {
            pushUndo();
            k.v = parseInt(sel.value);
            dirty = true;
            render();
          });
          row.textContent = (keypointNames[ki] || `kpt${ki}`) + " ";
          row.appendChild(sel);
          kDiv.appendChild(row);
        });
        const kInfo = document.createElement("p");
        kInfo.className = "muted";
        kInfo.style.fontSize = "11px";
        kInfo.textContent = "Drag keypoints to reposition";
        kDiv.appendChild(kInfo);
        fields.appendChild(kDiv);
      }
    } else {
      editor.style.display = "none";
    }
  }

  function deleteSelected() {
    if (selectedIdx >= 0 && selectedIdx < annotations.length) {
      pushUndo();
      annotations.splice(selectedIdx, 1);
      selectedIdx = -1;
      dirty = true;
      renderAnnotationList();
      render();
    }
  }

  // ── Render ─────────────────────────────────────────────
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    ctx.drawImage(img, 0, 0);

    if (task === "semantic_segmentation" && maskCanvas) {
      ctx.globalAlpha = 0.4;
      ctx.drawImage(maskCanvas, 0, 0);
      ctx.globalAlpha = 1.0;
    }

    annotations.forEach((ann, i) => drawAnnotation(ann, i === selectedIdx, i));

    if (drawing) {
      if (drawing.mode === "bbox") {
        const x = Math.min(drawing.x0, drawing.x1);
        const y = Math.min(drawing.y0, drawing.y1);
        const w = Math.abs(drawing.x1 - drawing.x0);
        const h = Math.abs(drawing.y1 - drawing.y0);
        ctx.strokeStyle = DRAWING_COLOR;
        ctx.lineWidth = 2 / zoom;
        ctx.setLineDash([6 / zoom, 3 / zoom]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
      } else if (drawing.mode === "polygon" && drawing.points.length > 0) {
        ctx.beginPath();
        drawing.points.forEach(([x, y], j) => {
          j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = DRAWING_COLOR;
        ctx.lineWidth = 2 / zoom;
        ctx.stroke();
        drawing.points.forEach(([x, y], j) => {
          ctx.beginPath();
          ctx.arc(x, y, 4 / zoom, 0, Math.PI * 2);
          ctx.fillStyle = j === 0 ? "#ff4444" : DRAWING_COLOR;
          ctx.fill();
        });
      } else if (drawing.mode === "keypoint") {
        drawing.keypoints.forEach((k, ki) => {
          ctx.beginPath();
          ctx.arc(k.x, k.y, 5 / zoom, 0, Math.PI * 2);
          ctx.fillStyle = BASE_PALETTE[ki % BASE_PALETTE.length];
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1 / zoom;
          ctx.stroke();
          ctx.fillStyle = "#fff";
          ctx.font = `${10 / zoom}px sans-serif`;
          ctx.fillText(keypointNames[ki] || ki, k.x + 6 / zoom, k.y - 6 / zoom);
        });
        skeletonEdges.forEach(([a, b]) => {
          if (a < drawing.keypoints.length && b < drawing.keypoints.length) {
            const ka = drawing.keypoints[a], kb = drawing.keypoints[b];
            ctx.beginPath();
            ctx.moveTo(ka.x, ka.y);
            ctx.lineTo(kb.x, kb.y);
            ctx.strokeStyle = "rgba(255,255,255,0.7)";
            ctx.lineWidth = 2 / zoom;
            ctx.stroke();
          }
        });
      }
    }

    ctx.restore();

    // HUD overlays (drawn outside the transform)
    if (drawing && drawing.mode === "keypoint") {
      const nextIdx = drawing.keypoints.length;
      if (nextIdx < keypointNames.length) {
        ctx.fillStyle = DRAWING_COLOR;
        ctx.font = "14px sans-serif";
        ctx.fillText(`Click to place: ${keypointNames[nextIdx]} (${nextIdx + 1}/${keypointNames.length})`, 10, canvas.height - 10);
      }
    }
    if (drawing && drawing.mode === "polygon") {
      ctx.fillStyle = DRAWING_COLOR;
      ctx.font = "13px sans-serif";
      const n = drawing.points.length;
      ctx.fillText(`Polygon: ${n} pts — right-click or dbl-click to close (min 3)`, 10, canvas.height - 10);
    }
  }

  function drawAnnotation(ann, selected, idx) {
    const color = colorForLabels(ann.labels);
    const lw = (selected ? 3 : 2) / zoom;

    if (ann.bbox && !ann.keypoints) {
      const [cx, cy, w, h] = ann.bbox;
      const [x0, y0] = normToImage(cx - w / 2, cy - h / 2);
      const [x1, y1] = normToImage(cx + w / 2, cy + h / 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = color;
      ctx.font = `bold ${12 / zoom}px sans-serif`;
      ctx.fillText(Object.values(ann.labels).join(", "), x0, y0 - 3 / zoom);
      ctx.fillStyle = color + "22";
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }

    if (ann.polygon) {
      const pts = ann.polygon.map(([nx, ny]) => normToImage(nx, ny));
      ctx.beginPath();
      pts.forEach(([x, y], j) => j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.closePath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.stroke();
      ctx.fillStyle = color + "33";
      ctx.fill();
      pts.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, (selected ? 4 : 2.5) / zoom, 0, Math.PI * 2);
        ctx.fillStyle = selected ? color : color + "AA";
        ctx.fill();
        if (selected) {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1 / zoom;
          ctx.stroke();
        }
      });
      ctx.fillStyle = color;
      ctx.font = `bold ${12 / zoom}px sans-serif`;
      ctx.fillText(Object.values(ann.labels).join(", "), pts[0][0], pts[0][1] - 3 / zoom);
    }

    if (ann.keypoints) {
      skeletonEdges.forEach(([a, b]) => {
        if (a < ann.keypoints.length && b < ann.keypoints.length) {
          const ka = ann.keypoints[a], kb = ann.keypoints[b];
          if (ka.v > 0 && kb.v > 0) {
            const [ax, ay] = normToImage(ka.x, ka.y);
            const [bx, by] = normToImage(kb.x, kb.y);
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2 / zoom;
            ctx.stroke();
          }
        }
      });
      ann.keypoints.forEach((k, ki) => {
        if (k.v === 0) return;
        const [kx, ky] = normToImage(k.x, k.y);
        ctx.beginPath();
        ctx.arc(kx, ky, (selected ? 6 : 5) / zoom, 0, Math.PI * 2);
        ctx.fillStyle = k.v === 1 ? "#999" : BASE_PALETTE[ki % BASE_PALETTE.length];
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5 / zoom;
        ctx.stroke();
        if (selected) {
          ctx.fillStyle = "#fff";
          ctx.font = `${9 / zoom}px sans-serif`;
          ctx.fillText(keypointNames[ki] || ki, kx + 7 / zoom, ky - 5 / zoom);
        }
      });
      if (ann.bbox) {
        const [cx, cy, w, h] = ann.bbox;
        const [x0, y0] = normToImage(cx - w / 2, cy - h / 2);
        const [x1, y1] = normToImage(cx + w / 2, cy + h / 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1 / zoom;
        ctx.setLineDash([4 / zoom, 4 / zoom]);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
      }
      ctx.fillStyle = color;
      ctx.font = `bold ${12 / zoom}px sans-serif`;
      const [lx, ly] = ann.bbox ? normToImage(ann.bbox[0] - ann.bbox[2] / 2, ann.bbox[1] - ann.bbox[3] / 2) : [0, 0];
      ctx.fillText(Object.values(ann.labels).join(", "), lx, ly - 3 / zoom);
    }
  }

  // ── Load / Save ────────────────────────────────────────
  async function loadAnnotation(filename) {
    const res = await fetch(`/api/project/${PID}/annotation/${encodeURIComponent(filename)}`);
    const data = await res.json();
    annotations = data.annotations || [];
    selectedIdx = -1;
    dirty = false;
    undoStack = [];
    redoStack = [];
    updateUndoButtons();

    if (task === "semantic_segmentation" && data.mask && maskCtx) {
      const maskImg = new Image();
      maskImg.onload = () => {
        maskCtx.clearRect(0, 0, imgW, imgH);
        maskCtx.drawImage(maskImg, 0, 0);
        render();
      };
      maskImg.src = "data:image/png;base64," + data.mask;
    }

    renderAnnotationList();
    render();
  }

  async function save() {
    let body;
    if (task === "semantic_segmentation") {
      const dataUrl = maskCanvas.toDataURL("image/png");
      body = { mask: dataUrl.split(",")[1] };
    } else {
      body = { annotations };
    }
    await fetch(`/api/project/${PID}/annotation/${encodeURIComponent(CURRENT)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    dirty = false;
    const dot = document.getElementById("dot-" + CURRENT_IDX);
    if (dot) dot.classList.add("labeled");
    const btn = document.getElementById("btnSave");
    btn.textContent = "✓ Saved";
    setTimeout(() => { btn.textContent = "💾 Save"; }, 1200);
  }

  // ── Boot ───────────────────────────────────────────────
  init();
})();
