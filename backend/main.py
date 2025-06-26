import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .utils.image_processing import analyze_images
from .utils.pdf_report import generate_pdf_report
from .models import AnalyzeResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

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
    os.makedirs("backend/static/results", exist_ok=True)
    # Save master image
    master_path = f"backend/static/results/{uuid.uuid4()}_{master_image.filename}"
    with open(master_path, "wb") as f:
        f.write(await master_image.read())
    # Process each comparison image
    results, processed_paths = await analyze_images(
        master_path, comparison_images, algorithm, sensitivity
    )
    # Generate PDF report
    pdf_path = generate_pdf_report(master_path, processed_paths, results)
    elapsed = time.time() - start_time
    # Build response
    return {
        "overall_similarity": results["overall_similarity"],
        "processing_time": elapsed,
        "comparisons": results["comparisons"],
        "pdf_url": f"/static/results/{os.path.basename(pdf_path)}"
    }

@app.get("/export/{filename}")
async def export(filename: str):
    file_path = f"backend/static/results/{filename}"
    return FileResponse(file_path)

@app.get("/ping")
async def ping():
    return {"status": "ok"} 