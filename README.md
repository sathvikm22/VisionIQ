# VisionIQ: Advanced Image Comparison

## Overview
VisionIQ is an advanced image comparison tool with a modern desktop UI and a FastAPI backend. It supports multiple algorithms, visual difference highlighting, and PDF report export.

---

## 1. Start the Backend (FastAPI)

**a.** Open a terminal and navigate to the `backend` directory:
```sh
cd /Users/sathvikmadhyastha/Projects/VisionIQ/backend
```

**b.** (Optional but recommended) Create and activate a virtual environment:
```sh
python3 -m venv venv
source venv/bin/activate
```

**c.** Install the backend dependencies:
```sh
pip install -r requirements.txt
```

**d.** Start the FastAPI server:
```sh
uvicorn main:app --reload
```
- The backend will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- You can check the health at: [http://127.0.0.1:8000/ping](http://127.0.0.1:8000/ping)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2. Run the Desktop App

**a.** Open a new terminal window/tab.

**b.** Navigate to your project root (where `VisionIQ_desktop.py` is located):
```sh
cd /Users/sathvikmadhyastha/Projects/VisionIQ
```

**c.** (Optional but recommended) Activate the same or a new virtual environment:
```sh
source backend/venv/bin/activate
```
*(or create a new one in the root if you prefer)*

**d.** Install PyQt5 and requests if not already installed:
```sh
pip install PyQt5 requests
```

**e.** Run the desktop app:
```sh
python VisionIQ_desktop.py
```

---

## 3. Usage
- **Upload a master image and one or more comparison images.**
- **Select algorithm and sensitivity.**
- **Click "Start Analysis".**
- **View results and export PDF if desired.**

---

## 4. Packaging as .exe (for Windows)
When you want to distribute as an executable:
- Use [PyInstaller](https://pyinstaller.org/) or [cx_Freeze](https://cx-freeze.readthedocs.io/).
- Make sure the backend is bundled or instructions are provided for running it.

---

## 5. Troubleshooting

### ModuleNotFoundError: No module named 'backend'
If you see an error like this when running the backend:
```
ModuleNotFoundError: No module named 'backend'
```
**Solution:**
- Make sure you are running `uvicorn` from the `backend` directory, and that the imports in `main.py` use relative imports (e.g., `from .utils.image_processing import analyze_images`).
- If you still see this error, double-check that your directory structure matches the repository and that you are using the correct Python environment.

---

## 6. Support
If you run into any errors, let us know the details and we'll help you debug! 