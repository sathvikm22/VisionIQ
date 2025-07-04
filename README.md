# VisionIQ: Advanced Image Comparison

## Overview
VisionIQ is an advanced image comparison tool with a modern desktop UI (PyQt5) and a FastAPI backend. It supports multiple algorithms (SSIM, MSE, PSNR, Histogram, Feature Matching, Deep Learning), visual difference highlighting, and PDF report export.

---

## 1. Getting Started

### a. Clone the Repository
```sh
git clone https://github.com/yourusername/visioniq.git
cd visioniq
```

### b. (Recommended) Set Up a Python Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Backend Setup (FastAPI)

### a. Navigate to the backend directory:
```sh
cd backend
```

### b. Create and activate a backend virtual environment (optional but recommended):
```sh
python3 -m venv venv
source venv/bin/activate
```

### c. Install backend dependencies:
```sh
pip install -r requirements.txt
# For best deep learning support, also run:
pip install git+https://github.com/openai/CLIP.git
```

### d. Download Model Weights (if not already present)
- Place required `.pth` or `.pt` model files in `backend/models/` as needed.
- For signature verification, see `backend/models/signet-pytorch/README.md` for dataset and training instructions.

### e. Start the FastAPI server:
```sh
uvicorn main:app --reload
```
- The backend will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Health check: [http://127.0.0.1:8000/ping](http://127.0.0.1:8000/ping)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 3. Desktop App Setup

### a. Open a new terminal window/tab.

### b. Navigate to your project root:
```sh
cd /path/to/your/project/visioniq
```

### c. (Optional) Activate the same or a new virtual environment:
```sh
source backend/venv/bin/activate
```

### d. Install desktop dependencies:
```sh
pip install PyQt5 requests
```

### e. Run the desktop app:
```sh
python VisionIQ_desktop.py
```

---

## 4. Usage

- Upload a master image and one or more comparison images.
- Select the algorithm and sensitivity.
- Click "Analysis" to compare.
- View results (side-by-side images, similarity, differences, algorithm-specific visuals).
- Download a PDF report if desired.

---

## 5. Packaging as .exe (for Windows)
- Use [PyInstaller](https://pyinstaller.org/) or [cx_Freeze](https://cx-freeze.readthedocs.io/).
- Ensure the backend is bundled or provide instructions for running it.

---

## 6. Troubleshooting

### "Image not found" in UI
- Make sure the backend is running and accessible at `http://127.0.0.1:8000`.
- Ensure all dependencies are installed and the backend has permission to write to `static/results/`.

### ModuleNotFoundError: No module named 'backend'
- Run `uvicorn` from the `backend` directory.
- Ensure relative imports are used in `main.py` (e.g., `from .utils.image_processing import analyze_images`).

---

## 7. Support

If you run into any errors, open an issue or contact the maintainer with details and logs.

---

**Notes:**
- For best results with deep learning, a GPU is recommended but not required.
- All output images and reports are saved in `static/results/`.
- Do not commit model weights, virtual environments, or build/dist artifacts to the repository.
- For signature verification, see `backend/models/signet-pytorch/README.md` for dataset and training instructions. 