import React, { useEffect, useState, useRef, useCallback } from "react";
import { Pipe } from "./types";
import * as api from "./api";
import "./App.css";

/** Compute 4 rectangle corners from centerline + width */
function pipeCorners(p: Pipe) {
  const dx = p.x2 - p.x1, dy = p.y2 - p.y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const nx = (-dy / len) * p.width / 2;
  const ny = (dx / len) * p.width / 2;
  return {
    c1x: p.x1 + nx, c1y: p.y1 + ny,
    c2x: p.x2 + nx, c2y: p.y2 + ny,
    c3x: p.x2 - nx, c3y: p.y2 - ny,
    c4x: p.x1 - nx, c4y: p.y1 - ny,
  };
}

function App() {
  const [pipes, setPipes] = useState<Pipe[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [scale, setScale] = useState(1);
  const [loading, setLoading] = useState(false);
  const [drawMode, setDrawMode] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [scaleText, setScaleText] = useState("");
  const [summary, setSummary] = useState<{ total_px: number; total_real: string } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getPipes().then(setPipes).catch(() => {});
    api.getScale().then((s) => setScaleText(s.display || s.raw_text || "")).catch(() => {});
    api.getSummary().then(setSummary).catch(() => {});
  }, []);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
      if (containerRef.current) {
        const cw = containerRef.current.clientWidth;
        setScale(Math.min(1, cw / img.naturalWidth));
      }
    };
    img.src = api.pageImageUrl + "?t=" + Date.now();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgSize.w) return;

    const dpr = window.devicePixelRatio || 1;
    const dw = imgSize.w * scale;
    const dh = imgSize.h * scale;
    canvas.width = dw * dpr;
    canvas.height = dh * dpr;
    canvas.style.width = dw + "px";
    canvas.style.height = dh + "px";
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, dw, dh);

    for (let i = 0; i < pipes.length; i++) {
      const p = pipes[i];
      const isSelected = p.id === selected;
      const c = pipeCorners(p);

      // Draw rectangle from 4 corners
      ctx.strokeStyle = isSelected ? "#ff4444" : "rgba(255,100,0,0.8)";
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(c.c1x * scale, c.c1y * scale);
      ctx.lineTo(c.c2x * scale, c.c2y * scale);
      ctx.lineTo(c.c3x * scale, c.c3y * scale);
      ctx.lineTo(c.c4x * scale, c.c4y * scale);
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = isSelected ? "rgba(255,68,68,0.2)" : "rgba(255,100,0,0.12)";
      ctx.fill();

      // Label
      const mx = ((p.x1 + p.x2) / 2) * scale;
      const my = ((p.y1 + p.y2) / 2) * scale;
      const label = String(i + 1);
      const fontSize = Math.max(10, 13 * scale);
      ctx.font = `bold ${fontSize}px sans-serif`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = isSelected ? "#ff4444" : "#ff6400";
      ctx.beginPath();
      ctx.roundRect(mx - tw / 2 - 4, my - fontSize / 2 - 3, tw + 8, fontSize + 6, 3);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, mx, my);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }

    if (drawStart) {
      ctx.strokeStyle = "#00ff00";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(drawStart.x * scale, drawStart.y * scale, 0, 0);
      ctx.setLineDash([]);
    }
  }, [pipes, selected, imgSize, scale, drawStart]);

  useEffect(() => { draw(); }, [draw]);

  // Mouse wheel zoom — zooms toward cursor position
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newScale = Math.min(3, Math.max(0.05, scale * factor));
      if (newScale === scale) return;

      // Mouse position relative to container viewport
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // Image coordinate under cursor
      const imgX = (container.scrollLeft + mx) / scale;
      const imgY = (container.scrollTop + my) / scale;

      setScale(newScale);

      // After scale change, adjust scroll so the same image point stays under cursor
      requestAnimationFrame(() => {
        container.scrollLeft = imgX * newScale - mx;
        container.scrollTop = imgY * newScale - my;
      });
    };
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, [scale]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;

    if (drawMode) {
      if (!drawStart) {
        setDrawStart({ x: Math.round(x), y: Math.round(y) });
      } else {
        const x1 = drawStart.x, y1 = drawStart.y;
        const x2 = Math.round(x), y2 = Math.round(y);
        const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        if (len < 20) { setDrawStart(null); return; }
        api.createPipe({ x1, y1, x2, y2, width: 30 }).then((p) => {
          setPipes((prev) => [...prev, p]);
          setSelected(p.id);
        });
        setDrawStart(null);
        setDrawMode(false);
      }
      return;
    }

    let best: string | null = null;
    let bestDist = 30 / scale;
    for (const p of pipes) {
      const cx = (p.x1 + p.x2) / 2;
      const cy = (p.y1 + p.y2) / 2;
      const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (d < bestDist) { bestDist = d; best = p.id; }
    }
    setSelected(best);
  };

  const handleDetect = async () => {
    setLoading(true);
    try {
      const result = await api.detectPipes();
      const p = await api.getPipes();
      setPipes(p);
      const s = await api.getSummary();
      setSummary(s);
      if ((result as any).scale?.display) setScaleText((result as any).scale.display);
    } finally { setLoading(false); }
  };

  const handleDelete = async () => {
    if (!selected) return;
    await api.deletePipe(selected);
    setPipes((prev) => prev.filter((p) => p.id !== selected));
    setSelected(null);
  };

  const handleSave = () => api.saveAll();
  const selectedPipe = pipes.find((p) => p.id === selected);

  const handleLengthChange = async (newLength: number) => {
    if (!selectedPipe) return;
    const oldLen = selectedPipe.length_px;
    if (oldLen < 1) return;
    const ratio = newLength / oldLen;
    const cx = (selectedPipe.x1 + selectedPipe.x2) / 2;
    const cy = (selectedPipe.y1 + selectedPipe.y2) / 2;
    const updated = await api.updatePipe(selectedPipe.id, {
      x1: Math.round(cx + (selectedPipe.x1 - cx) * ratio),
      y1: Math.round(cy + (selectedPipe.y1 - cy) * ratio),
      x2: Math.round(cx + (selectedPipe.x2 - cx) * ratio),
      y2: Math.round(cy + (selectedPipe.y2 - cy) * ratio),
    });
    setPipes((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
  };

  const handleCoordChange = async (field: "x1" | "y1" | "x2" | "y2" | "width", value: number) => {
    if (!selectedPipe) return;
    const updated = await api.updatePipe(selectedPipe.id, {
      x1: field === "x1" ? value : selectedPipe.x1,
      y1: field === "y1" ? value : selectedPipe.y1,
      x2: field === "x2" ? value : selectedPipe.x2,
      y2: field === "y2" ? value : selectedPipe.y2,
      width: field === "width" ? value : selectedPipe.width,
    });
    setPipes((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
  };

  const zoomToCenter = (factor: number) => {
    const container = containerRef.current;
    if (!container) return;
    const newScale = Math.min(3, Math.max(0.05, scale * factor));
    if (newScale === scale) return;
    const cx = container.scrollLeft + container.clientWidth / 2;
    const cy = container.scrollTop + container.clientHeight / 2;
    const imgX = cx / scale;
    const imgY = cy / scale;
    setScale(newScale);
    requestAnimationFrame(() => {
      container.scrollLeft = imgX * newScale - container.clientWidth / 2;
      container.scrollTop = imgY * newScale - container.clientHeight / 2;
    });
  };
  const zoomIn = () => zoomToCenter(1.2);
  const zoomOut = () => zoomToCenter(1 / 1.2);

  return (
    <div className="app">
      <div className="toolbar">
        <div className="app-title">
          <span className="app-name">DuctVision</span>
          <span className="app-desc">HVAC Duct Detection & Measurement Tool</span>
        </div>
        <span className="separator" />
        <button onClick={handleDetect} disabled={loading}>
          {loading ? "Detecting..." : "🔍 Detect Pipes"}
        </button>
        <button onClick={() => { setDrawMode(!drawMode); setDrawStart(null); }}
                className={drawMode ? "active" : ""}>
          {drawMode ? "✏️ Drawing..." : "➕ Add Pipe"}
        </button>
        <button onClick={handleSave}>💾 Save</button>
        <span className="separator" />
        <button onClick={zoomOut}>−</button>
        <span className="zoom-label">{Math.round(scale * 100)}%</span>
        <button onClick={zoomIn}>+</button>
        <span className="pipe-count">{pipes.length} pipes</span>
      </div>

      <div className="main">
        <div className="canvas-container" ref={containerRef}>
          <canvas ref={canvasRef} onClick={handleCanvasClick}
                  style={{ cursor: drawMode ? "crosshair" : "pointer" }} />
        </div>

        <div className="sidebar">
          {(scaleText || summary) && (
            <div className="scale-info">
              {scaleText && <div className="scale-text">📐 Scale: {scaleText}</div>}
              {summary && (
                <>
                  <div className="summary-row">
                    <span>Total length (px)</span>
                    <span>{Math.round(summary.total_px).toLocaleString()}</span>
                  </div>
                  <div className="summary-row">
                    <span>Total length (real)</span>
                    <span>{summary.total_real}</span>
                  </div>
                </>
              )}
            </div>
          )}
          <h3>Pipe Details</h3>
          {selectedPipe ? (
            <div className="pipe-detail">
              <div className="field">
                <label>ID</label>
                <span>{selectedPipe.id}</span>
              </div>
              <div className="field">
                <label>Source</label>
                <span>{selectedPipe.source}</span>
              </div>
              <div className="field">
                <label>Length (px)</label>
                <input type="number" value={Math.round(selectedPipe.length_px)}
                  onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) handleLengthChange(v); }} />
              </div>
              <div className="field">
                <label>Angle (°)</label>
                <input type="number" step="0.5" value={selectedPipe.angle}
                  onChange={(e) => {
                    const newAngle = parseFloat(e.target.value);
                    if (isNaN(newAngle)) return;
                    const delta = ((newAngle - selectedPipe.angle) * Math.PI) / 180;
                    const cx = (selectedPipe.x1 + selectedPipe.x2) / 2;
                    const cy = (selectedPipe.y1 + selectedPipe.y2) / 2;
                    const rot = (px: number, py: number) => ({
                      x: Math.round(cx + (px - cx) * Math.cos(delta) - (py - cy) * Math.sin(delta)),
                      y: Math.round(cy + (px - cx) * Math.sin(delta) + (py - cy) * Math.cos(delta)),
                    });
                    const p1 = rot(selectedPipe.x1, selectedPipe.y1);
                    const p2 = rot(selectedPipe.x2, selectedPipe.y2);
                    api.updatePipe(selectedPipe.id, { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y })
                      .then((u) => setPipes((prev) => prev.map((p) => (p.id === u.id ? u : p))));
                  }} />
              </div>
              <div className="field">
                <label>Width (px)</label>
                <input type="number" value={Math.round(selectedPipe.width)}
                  onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) handleCoordChange("width", v); }} />
              </div>
              <div className="coord-group">
                <label>Centerline</label>
                <div className="coord-row">
                  <span>x1</span><input type="number" value={selectedPipe.x1} onChange={(e) => handleCoordChange("x1", parseInt(e.target.value) || 0)} />
                  <span>y1</span><input type="number" value={selectedPipe.y1} onChange={(e) => handleCoordChange("y1", parseInt(e.target.value) || 0)} />
                </div>
                <div className="coord-row">
                  <span>x2</span><input type="number" value={selectedPipe.x2} onChange={(e) => handleCoordChange("x2", parseInt(e.target.value) || 0)} />
                  <span>y2</span><input type="number" value={selectedPipe.y2} onChange={(e) => handleCoordChange("y2", parseInt(e.target.value) || 0)} />
                </div>
              </div>
              <button className="delete-btn" onClick={handleDelete}>🗑️ Delete Pipe</button>
            </div>
          ) : (
            <p className="hint">Click a pipe on the drawing to select it</p>
          )}

          <h3>All Pipes</h3>
          <div className="pipe-list">
            {pipes.map((p, i) => (
              <div key={p.id} className={`pipe-item ${p.id === selected ? "selected" : ""}`}
                   onClick={() => setSelected(p.id)}>
                <span className="pipe-num">{i + 1}</span>
                <span className="pipe-id">{p.id}</span>
                <span className="pipe-len">{Math.round(p.length_px)}px</span>
                <span className="pipe-src">{p.source}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
