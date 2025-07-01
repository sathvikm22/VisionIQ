import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .utils.image_processing import analyze_images
from .utils.pdf_report import generate_pdf_report
from .models import AnalyzeResponse
import urllib.parse

app = FastAPI()
# Mount static files from the root static directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    master_image: UploadFile = File(...),
    comparison_images: list[UploadFile] = File(...),
    algorithm: str = Form(...),
    sensitivity: int = Form(...),
):
    """
    Analyze comparison images against a master image using the selected algorithm and sensitivity.
    Returns similarity scores, difference counts, processed image URLs, unique visual outputs, and a PDF report URL.
    """
    start_time = time.time()
    os.makedirs("static/results", exist_ok=True)
    # Save master image
    master_path = f"static/results/{uuid.uuid4()}_{master_image.filename}"
    with open(master_path, "wb") as f:
        f.write(await master_image.read())
    # Process each comparison image
    results, processed_paths = await analyze_images(
        master_path, comparison_images, algorithm, sensitivity
    )
    # Generate PDF report
    pdf_path = generate_pdf_report(master_path, processed_paths, results)
    # Sanitize PDF URL for response
    pdf_url = f"/static/results/{urllib.parse.quote(os.path.basename(pdf_path))}"
    elapsed = time.time() - start_time
    # Build response
    return {
        "overall_similarity": results["overall_similarity"],
        "processing_time": elapsed,
        "comparisons": results["comparisons"],
        "pdf_url": pdf_url
    }

@app.get("/export/{filename}")
async def export(filename: str):
    file_path = f"static/results/{filename}"
    return FileResponse(file_path)

@app.get("/ping")
async def ping():
    return {"status": "ok"} 