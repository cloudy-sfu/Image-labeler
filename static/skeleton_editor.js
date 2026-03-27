(function () {
  function createSkeletonEditor(config) {
    const canvas = document.getElementById(config.canvasId);
    const addVertexBtn = document.getElementById(config.addVertexBtnId);
    const selectToolBtn = document.getElementById(config.selectToolBtnId);
    const addEdgeToolBtn = document.getElementById(config.addEdgeToolBtnId);
    const deleteEdgeToolBtn = document.getElementById(config.deleteEdgeToolBtnId);
    const vertexNameInput = document.getElementById(config.vertexNameInputId);
    const deleteVertexBtn = document.getElementById(config.deleteVertexBtnId);

    if (!canvas) {
      return null;
    }

    const ctx = canvas.getContext("2d");
    const state = {
      vertices: [],
      edges: [],
      selectedVertex: -1,
      mode: "select",
      pendingEdgeVertex: -1,
      dragVertex: -1,
      dragStartPos: null,
      isPanDrag: false,
    };

    function sortedEdge(a, b) {
      return a < b ? [a, b] : [b, a];
    }

    function edgeKey(a, b) {
      const [u, v] = sortedEdge(a, b);
      return `${u}-${v}`;
    }

    function edgeExists(a, b) {
      return state.edges.some(([u, v]) => {
        const x = Math.min(u, v);
        const y = Math.max(u, v);
        const [m, n] = sortedEdge(a, b);
        return x === m && y === n;
      });
    }

    function addEdge(a, b) {
      if (a === b || a < 0 || b < 0 || a >= state.vertices.length || b >= state.vertices.length) {
        return;
      }
      if (edgeExists(a, b)) {
        return;
      }
      state.edges.push(sortedEdge(a, b));
    }

    function removeEdge(a, b) {
      const target = edgeKey(a, b);
      state.edges = state.edges.filter(([u, v]) => edgeKey(u, v) !== target);
    }

    function setMode(mode) {
      state.mode = mode;
      state.pendingEdgeVertex = -1;
      if (selectToolBtn) selectToolBtn.classList.toggle("active", mode === "select");
      if (addEdgeToolBtn) addEdgeToolBtn.classList.toggle("active", mode === "add_edge");
      if (deleteEdgeToolBtn) deleteEdgeToolBtn.classList.toggle("active", mode === "delete_edge");
      draw();
    }

    function getCanvasPos(e) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    }

    function hitVertex(pos) {
      const radius = 14;
      for (let i = state.vertices.length - 1; i >= 0; i -= 1) {
        const v = state.vertices[i];
        const dx = pos.x - v.x;
        const dy = pos.y - v.y;
        if ((dx * dx + dy * dy) <= radius * radius) {
          return i;
        }
      }
      return -1;
    }

    function updateSelectedVertexUI() {
      const selected = state.selectedVertex;
      if (selected >= 0 && selected < state.vertices.length) {
        vertexNameInput.disabled = false;
        deleteVertexBtn.disabled = false;
        vertexNameInput.value = state.vertices[selected].name || "";
      } else {
        vertexNameInput.disabled = true;
        deleteVertexBtn.disabled = true;
        vertexNameInput.value = "";
      }
    }

    function setSelectedVertex(idx) {
      state.selectedVertex = idx;
      updateSelectedVertexUI();
      draw();
    }

    function addVertex() {
      const n = state.vertices.length;
      const centerX = canvas.clientWidth / 2;
      const centerY = canvas.clientHeight / 2;
      const radius = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.3;
      const angle = (Math.PI * 2 * n) / Math.max(6, n + 1);
      state.vertices.push({
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        name: `keypoint_${n}`,
      });
      setSelectedVertex(n);
    }

    function deleteVertex(idx) {
      if (idx < 0 || idx >= state.vertices.length) {
        return;
      }

      state.vertices.splice(idx, 1);

      // Reindex endpoints so vertex IDs stay contiguous after deletion.
      const dedupe = new Set();
      const remapped = [];
      state.edges.forEach(([a, b]) => {
        if (a === idx || b === idx) {
          return;
        }
        const na = a > idx ? a - 1 : a;
        const nb = b > idx ? b - 1 : b;
        const [u, v] = sortedEdge(na, nb);
        const key = edgeKey(u, v);
        if (!dedupe.has(key)) {
          dedupe.add(key);
          remapped.push([u, v]);
        }
      });
      state.edges = remapped;

      if (state.selectedVertex === idx) {
        state.selectedVertex = -1;
      } else if (state.selectedVertex > idx) {
        state.selectedVertex -= 1;
      }

      if (state.pendingEdgeVertex === idx) {
        state.pendingEdgeVertex = -1;
      } else if (state.pendingEdgeVertex > idx) {
        state.pendingEdgeVertex -= 1;
      }

      updateSelectedVertexUI();
      draw();
    }

    function handleEdgeToolClick(vertexIdx) {
      if (state.pendingEdgeVertex < 0) {
        // First click: select starting point
        state.pendingEdgeVertex = vertexIdx;
        setSelectedVertex(vertexIdx);
        draw();
        return;
      }

      const a = state.pendingEdgeVertex;
      const b = vertexIdx;
      
      // Second click: complete the edge pair
      if (a !== b) {
        if (state.mode === "add_edge") {
          addEdge(a, b);
        } else if (state.mode === "delete_edge") {
          removeEdge(a, b);
        }
      }
      
      // Reset: no pending vertex, no selection
      state.pendingEdgeVertex = -1;
      state.selectedVertex = -1;
      updateSelectedVertexUI();
      draw();
    }

    function clampVertex(v) {
      v.x = Math.max(14, Math.min(canvas.clientWidth - 14, v.x));
      v.y = Math.max(14, Math.min(canvas.clientHeight - 14, v.y));
    }

    function onCanvasPointerDown(e) {
      const pos = getCanvasPos(e);
      const hit = hitVertex(pos);

      if (state.mode === "add_edge" || state.mode === "delete_edge") {
        if (hit >= 0) {
          handleEdgeToolClick(hit);
        }
        return;
      }

      if (hit >= 0) {
        setSelectedVertex(hit);
        state.dragVertex = hit;
        state.isPanDrag = false;
      } else {
        setSelectedVertex(-1);
        state.dragStartPos = pos;
        state.isPanDrag = true;
      }
    }

    function onCanvasPointerMove(e) {
      const pos = getCanvasPos(e);

      if (state.isPanDrag && state.dragStartPos) {
        const dx = pos.x - state.dragStartPos.x;
        const dy = pos.y - state.dragStartPos.y;
        state.vertices.forEach(v => {
          v.x += dx;
          v.y += dy;
          clampVertex(v);
        });
        state.dragStartPos = pos;
        draw();
        return;
      }

      if (state.dragVertex < 0) {
        return;
      }
      const v = state.vertices[state.dragVertex];
      if (!v) {
        return;
      }
      v.x = pos.x;
      v.y = pos.y;
      clampVertex(v);
      draw();
    }

    function onCanvasPointerUp() {
      state.dragVertex = -1;
      state.isPanDrag = false;
      state.dragStartPos = null;
    }

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(200, Math.floor(canvas.clientWidth));
      const h = Math.max(180, Math.floor(canvas.clientHeight));
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      state.vertices.forEach(clampVertex);
      draw();
    }

    function draw() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = "#9fb4ff";
      ctx.lineWidth = 2;
      state.edges.forEach(([a, b]) => {
        const va = state.vertices[a];
        const vb = state.vertices[b];
        if (!va || !vb) {
          return;
        }
        ctx.beginPath();
        ctx.moveTo(va.x, va.y);
        ctx.lineTo(vb.x, vb.y);
        ctx.stroke();
      });

      if (state.pendingEdgeVertex >= 0 && state.vertices[state.pendingEdgeVertex]) {
        const v = state.vertices[state.pendingEdgeVertex];
        ctx.beginPath();
        ctx.arc(v.x, v.y, 18, 0, Math.PI * 2);
        ctx.strokeStyle = "#f39c12";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      state.vertices.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(v.x, v.y, 14, 0, Math.PI * 2);
        ctx.fillStyle = i === state.selectedVertex ? "#4a6cf7" : "#ffffff";
        ctx.strokeStyle = "#2f4cb2";
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = i === state.selectedVertex ? "#ffffff" : "#2f4cb2";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i), v.x, v.y);
      });
    }

    addVertexBtn.addEventListener("click", addVertex);
    selectToolBtn.addEventListener("click", () => setMode("select"));
    addEdgeToolBtn.addEventListener("click", () => setMode("add_edge"));
    deleteEdgeToolBtn.addEventListener("click", () => setMode("delete_edge"));
    deleteVertexBtn.addEventListener("click", () => {
      if (state.selectedVertex >= 0) {
        deleteVertex(state.selectedVertex);
      }
    });
    vertexNameInput.addEventListener("input", (e) => {
      if (state.selectedVertex < 0 || !state.vertices[state.selectedVertex]) {
        return;
      }
      state.vertices[state.selectedVertex].name = e.target.value;
    });

    canvas.addEventListener("pointerdown", onCanvasPointerDown);
    canvas.addEventListener("pointermove", onCanvasPointerMove);
    canvas.addEventListener("pointerup", onCanvasPointerUp);
    canvas.addEventListener("pointerleave", onCanvasPointerUp);
    window.addEventListener("resize", resizeCanvas);

    setMode("select");
    updateSelectedVertexUI();
    resizeCanvas();

    return {
      getKeypointNames() {
        return state.vertices.map((v, i) => {
          const name = (v.name || "").trim();
          return name || `keypoint_${i}`;
        });
      },
      getSkeletonEdges() {
        return state.edges.map(([a, b]) => [a, b]);
      },
      resize() {
        resizeCanvas();
      },
    };
  }

  window.createSkeletonEditor = createSkeletonEditor;
})();

