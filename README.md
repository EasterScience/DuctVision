# DuctVision

HVAC Duct Detection & Measurement Tool — automatically detects rectangular ducts/pipes in mechanical engineering drawings (PDF), measures their lengths, and provides a web-based viewer for review and editing.

## Features

- **Automatic duct detection** from PDF mechanical drawings using multi-layer computer vision (white channel analysis, diagonal pipe detection, Hough line pairs, black contour rectangles)
- **Scale extraction** via OCR — reads scale notations like `1/4" = 1'-0"` from the title block
- **Real-world measurements** — converts pixel lengths to feet-inches using the extracted scale
- **Interactive web viewer** — pan, zoom (mouse wheel toward cursor), click to select pipes
- **Edit pipes** — adjust centerline coordinates, width, length, and angle
- **Add/delete pipes** manually
- **Save/load** pipe data as JSON

## Tech Stack

- **Backend**: Python 3.12, FastAPI, OpenCV, NumPy, Tesseract OCR, PyMuPDF
- **Frontend**: React, TypeScript, HTML5 Canvas
- **Package manager**: uv (Python), npm (frontend)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Node.js 18+ and npm
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on PATH

## Quick Start

### Using the batch file (Windows)

```
run.bat
```

This starts both the backend (port 8000) and frontend (port 3000).

### Manual setup

1. Install Python dependencies:

```bash
uv sync
```

2. Install frontend dependencies:

```bash
cd frontend
npm install
```

3. Place your PDF in the `input/` folder (default: `input/testset2.pdf`).

4. Start the backend:

```bash
uv run uvicorn api:app --reload --port 8000
```

5. Start the frontend:

```bash
cd frontend
npm start
```

6. Open http://localhost:3000 and click **Detect Pipes**.

## Project Structure

```
├── api.py                 # FastAPI backend (REST endpoints)
├── pipe_marker.py         # Core duct detection engine
├── scale_extractor.py     # Drawing scale OCR & parsing
├── ocr_engine.py          # Tesseract OCR wrapper
├── pdf_renderer.py        # PDF to image rendering
├── models.py              # Data models
├── main.py                # CLI entry point
├── frontend/              # React TypeScript UI
│   ├── src/App.tsx        # Main viewer component
│   ├── src/api.ts         # API client
│   └── src/types.ts       # TypeScript interfaces
├── input/                 # Input PDF files
├── output/                # Detection results (JSON)
├── run.bat                # Windows launcher
├── Dockerfile             # Container build
└── docker-compose.yml     # Container orchestration
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Run duct detection on PDF |
| GET | `/api/page-image` | Get PDF page as JPEG |
| GET | `/api/pipes` | List all detected pipes |
| PUT | `/api/pipes/{id}` | Update pipe coordinates |
| POST | `/api/pipes` | Create new pipe |
| DELETE | `/api/pipes/{id}` | Delete a pipe |
| POST | `/api/save` | Save pipe data to disk |
| GET | `/api/scale` | Get extracted drawing scale |
| GET | `/api/summary` | Get total pipe length summary |

## Environment Variables

Create a `.env` file in the project root:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

## License

Private — all rights reserved.
