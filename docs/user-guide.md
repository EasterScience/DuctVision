# DuctVision — User Guide

This guide walks you through every feature of DuctVision, from loading a drawing to exporting your results.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Loading a Drawing](#loading-a-drawing)
4. [Running Detection](#running-detection)
5. [Navigating the Canvas](#navigating-the-canvas)
6. [Selecting Pipes](#selecting-pipes)
7. [Moving Pipes (Drag & Drop)](#moving-pipes-drag--drop)
8. [Resizing Pipes (Corner Handles)](#resizing-pipes-corner-handles)
9. [Adding a New Pipe](#adding-a-new-pipe)
10. [Editing Pipe Properties](#editing-pipe-properties)
11. [Deleting a Pipe](#deleting-a-pipe)
12. [Scale & Measurements](#scale--measurements)
13. [Saving Your Work](#saving-your-work)
14. [Keyboard & Mouse Reference](#keyboard--mouse-reference)
15. [Troubleshooting](#troubleshooting)

---

## Getting Started

1. Make sure the backend and frontend are running (see [README](../README.md) for setup).
2. Open your browser to **http://localhost:3000**.
3. You should see the toolbar at the top and a blank canvas area with a sidebar on the right.

---

## Interface Overview

The application has three main areas:

### Toolbar (top bar)

| Element | Description |
|---------|-------------|
| **DuctVision** title | App name and subtitle |
| 🔍 Detect Pipes | Runs automatic duct detection on the loaded PDF |
| ➕ Add Pipe | Enters draw mode to manually add a pipe |
| 💾 Save | Saves all pipe data to disk |
| − / + buttons | Zoom out / zoom in (centered on viewport) |
| Zoom % | Current zoom level |
| Pipe count | Total number of detected/added pipes |

### Canvas (center)

The main drawing area where the PDF page is rendered. All pipes are drawn as colored rectangles overlaid on the image. You can zoom, pan, select, drag, and resize pipes here.

### Sidebar (right)

- **Scale info** — Shows the extracted drawing scale and total pipe lengths (pixels and real-world units)
- **Pipe Details** — When a pipe is selected, shows all its properties with editable fields
- **All Pipes** — Scrollable list of every pipe; click to select

---

## Loading a Drawing

Place your PDF file in the `input/` folder. The default file is `input/testset2.pdf`. The backend automatically loads the first page when the app starts.

To change the input file, update the PDF in the `input/` folder and restart the backend, or modify the file path in the API configuration.

---

## Running Detection

1. Click **🔍 Detect Pipes** in the toolbar.
2. The button shows "Detecting..." while processing.
3. Once complete, detected pipes appear as orange rectangles on the canvas.
4. The sidebar updates with the pipe count and total length summary.
5. The drawing scale (if found) appears in the sidebar scale info section.

Detection uses multiple computer vision strategies:
- White channel analysis for light-colored ducts
- Diagonal pipe detection for angled ducts
- Hough line pair matching for parallel lines
- Black contour rectangle detection

---

## Navigating the Canvas

### Zooming

- **Mouse wheel** — Scroll up to zoom in, scroll down to zoom out. The zoom centers on your cursor position, so the point under your mouse stays fixed.
- **Toolbar +/−** — Zoom in/out centered on the middle of the visible area.
- The zoom level is displayed as a percentage in the toolbar (e.g., "45%").
- Zoom range: 5% to 300%.

### Panning

When the drawing is larger than the visible area, scroll bars appear. Click and drag the scroll bars, or use your mouse/trackpad to scroll horizontally and vertically within the canvas container.

---

## Selecting Pipes

- **On the canvas** — Click on any pipe rectangle. The selected pipe turns red with a semi-transparent red fill. If you click on empty space, the selection is cleared.
- **From the sidebar list** — Click any pipe in the "All Pipes" list at the bottom of the sidebar.

When a pipe is selected:
- It highlights red on the canvas
- Its details appear in the sidebar (ID, source, length, angle, width, coordinates)
- Four white corner handles appear at the rectangle corners for resizing

---

## Moving Pipes (Drag & Drop)

1. Make sure you are **not** in draw mode (the "Add Pipe" button should not show "Drawing...").
2. Click and hold on any pipe rectangle on the canvas.
3. Drag the mouse to move the pipe to a new position.
4. Release the mouse button to drop it.
5. The new position is automatically saved to the backend.
6. The summary updates to reflect any length changes.

The cursor changes to a grabbing hand while dragging.

---

## Resizing Pipes (Corner Handles)

1. **Select a pipe** by clicking on it (it turns red).
2. Four small **white circles with red borders** appear at the four corners of the rectangle.
3. Click and drag any corner handle to reshape the pipe:
   - Moving a corner changes the pipe's length, width, and/or angle.
   - The centerline and width are recalculated from the new corner positions.
4. Release the mouse to finalize. The updated geometry is saved to the backend.

Corner layout:
```
  Corner 1 ●───────────● Corner 2
           |             |
           |  centerline |
           |             |
  Corner 4 ●───────────● Corner 3
```

- Corners 1 & 4 are at the `(x1, y1)` end of the centerline
- Corners 2 & 3 are at the `(x2, y2)` end of the centerline

---

## Adding a New Pipe

1. Click **➕ Add Pipe** in the toolbar. The button changes to "✏️ Drawing..." and the cursor becomes a crosshair.
2. **Click** on the canvas to set the start point (a green dot appears).
3. **Move the mouse** — a green dashed rectangle preview follows your cursor, showing the pipe shape and its length in pixels.
4. **Click again** to place the pipe. The pipe is created with a default width of 30px.
5. Draw mode automatically exits after placing a pipe.
6. A "Pipe created" toast notification appears.
7. To cancel draw mode without placing a pipe, click the "✏️ Drawing..." button again.

If the distance between start and end points is less than 20px, the pipe is not created (prevents accidental tiny pipes).

---

## Editing Pipe Properties

When a pipe is selected, the sidebar shows editable fields:

### Length (px)
- Type a new pixel length in the input field.
- Click the **Update** button to apply. The pipe resizes symmetrically around its center.
- The length does not update live while typing — you must click Update.

### Angle (°)
- Type a new angle value. The pipe rotates around its center point.
- Changes apply immediately on input.

### Width (px)
- Type a new width value. The rectangle becomes wider or narrower.
- Changes apply immediately on input.

### Centerline Coordinates
- **x1, y1** — Start point of the centerline
- **x2, y2** — End point of the centerline
- Edit any coordinate directly. Changes apply immediately.

All edits are sent to the backend and the canvas redraws in real time.

---

## Deleting a Pipe

1. Select a pipe (click on canvas or sidebar list).
2. Click **🗑️ Delete Pipe** at the bottom of the pipe details section.
3. The pipe is removed from the canvas and backend.
4. A "Pipe deleted" toast notification appears.
5. The summary updates automatically.

---

## Scale & Measurements

The sidebar shows measurement information at the top:

- **📐 Scale** — The drawing scale extracted via OCR (e.g., `1/4" = 1'-0"`). This is read from the title block of the PDF.
- **Total length (px)** — Sum of all pipe lengths in pixels.
- **Total length (real)** — Sum converted to real-world units using the extracted scale (e.g., feet and inches).

These values update automatically whenever you add, delete, move, or resize pipes.

---

## Saving Your Work

- Click **💾 Save** in the toolbar to persist all pipe data to disk.
- Data is saved as JSON in the `output/` folder.
- A "Saved successfully" toast notification confirms the save.
- Pipe data is also auto-saved to the backend on every edit (move, resize, coordinate change), but the **Save** button writes the final state to disk files.

---

## Keyboard & Mouse Reference

| Action | Input |
|--------|-------|
| Zoom in/out | Mouse wheel on canvas |
| Zoom in (toolbar) | Click **+** button |
| Zoom out (toolbar) | Click **−** button |
| Select pipe | Left-click on pipe |
| Deselect | Left-click on empty canvas |
| Move pipe | Left-click + drag on pipe body |
| Resize pipe | Left-click + drag on corner handle |
| Add pipe (start) | Click **➕ Add Pipe**, then click on canvas |
| Add pipe (finish) | Click second point on canvas |
| Cancel draw mode | Click **✏️ Drawing...** button |

---

## Troubleshooting

### Pipes not detected
- Make sure Tesseract OCR is installed and on your system PATH.
- Check that the PDF is a mechanical/HVAC drawing with visible duct lines.
- Very low-contrast or colored drawings may need preprocessing.

### Scale not found
- The OCR looks for scale text in the title block area (bottom-right of the drawing).
- If the scale notation is non-standard, it may not be recognized. You can still work with pixel measurements.

### Canvas is blank
- Verify the backend is running on port 8000.
- Check the browser console for CORS or network errors.
- Make sure a PDF exists in the `input/` folder.

### Drag not working
- Make sure you are not in draw mode (button should say "➕ Add Pipe", not "✏️ Drawing...").
- Click directly on the pipe rectangle area, not just near it.

### Changes not persisting after restart
- Click **💾 Save** before closing the app. Edits are held in memory until saved to disk.
