from fpdf import FPDF
import os
from typing import List, Dict
from PIL import Image

def generate_pdf_report(master_path: str, processed_paths: List[str], results: Dict) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "VisionIQ Analysis Report", ln=True, align='C')
    pdf.ln(10)
    # Master image
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Master Image: {os.path.basename(master_path)}", ln=True)
    try:
        pdf.image(master_path, w=100)
    except:
        pass
    pdf.ln(10)
    # Table header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Filename", 1)
    pdf.cell(40, 10, "Similarity %", 1)
    pdf.cell(40, 10, "# Differences", 1)
    pdf.ln()
    pdf.set_font("Arial", '', 12)
    for comp in results["comparisons"]:
        pdf.cell(60, 10, comp["filename"], 1)
        pdf.cell(40, 10, str(comp["similarity_score"]), 1)
        pdf.cell(40, 10, str(comp["num_differences"]), 1)
        pdf.ln()
    pdf.ln(5)
    # Images with green boxes and unique visuals
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Comparison Images with Differences Highlighted and Unique Visuals:", ln=True)
    for comp in results["comparisons"]:
        # Green boxes
        try:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"{comp['filename']} - Differences Highlighted", ln=True)
            pdf.image(os.path.join("backend/static", comp["processed_image_url"].replace("static/", "")), w=100)
            pdf.ln(2)
        except:
            pass
        # Unique visual
        if comp.get("visual_output"):
            try:
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 7, f"{comp.get('visual_label', '')}", ln=True)
                pdf.image(os.path.join("backend/static", comp["visual_output"].replace("static/", "")), w=100)
                pdf.ln(5)
            except:
                continue
    # Overall stats
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Overall Similarity: {results['overall_similarity']}%", ln=True)
    pdf.ln(5)
    # Save PDF
    pdf_name = f"backend/static/results/visioniq_report_{os.path.basename(master_path)}.pdf"
    pdf.output(pdf_name)
    return pdf_name 