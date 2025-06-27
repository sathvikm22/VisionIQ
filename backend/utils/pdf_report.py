from fpdf import FPDF
import os
from typing import List, Dict
from PIL import Image

def generate_pdf_report(master_path: str, processed_paths: List[str], results: Dict) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 12, "VisionIQ Analysis Report", ln=True, align='C')
    pdf.ln(4)
    # Master image
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 10, f"Master Image: {os.path.basename(master_path)}", ln=True)
    try:
        if master_path and os.path.exists(master_path):
            pdf.image(master_path, w=80)
        else:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 8, "Image not found", ln=True)
    except Exception:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 8, "Image not found", ln=True)
    pdf.ln(6)
    # Table header
    pdf.set_font("Arial", 'B', 12)
    th = pdf.font_size + 3
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(60, th, "Filename", border=1, align='C', fill=True)
    pdf.cell(35, th, "Similarity %", border=1, align='C', fill=True)
    pdf.cell(35, th, "# Differences", border=1, align='C', fill=True)
    pdf.ln(th)
    pdf.set_font("Arial", '', 12)
    pdf.set_fill_color(255, 255, 255)
    for comp in results["comparisons"]:
        pdf.cell(60, th, comp["filename"], border=1, align='C', fill=True)
        pdf.cell(35, th, str(comp["similarity_score"]), border=1, align='C', fill=True)
        pdf.cell(35, th, str(comp["num_differences"]), border=1, align='C', fill=True)
        pdf.ln(th)
    pdf.ln(4)
    # Images with green boxes and unique visuals
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 10, "Comparison Images with Differences Highlighted and Visuals:", ln=True)
    pdf.ln(2)
    for comp in results["comparisons"]:
        # Green boxes
        try:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"{comp['filename']} - Differences Highlighted", ln=True)
            # Robust path resolution for green box image
            green_box_path = comp["processed_image_url"]
            if green_box_path.startswith("/static/"):
                green_box_path = green_box_path.replace("/static/", "backend/static/")
            elif green_box_path.startswith("static/"):
                green_box_path = green_box_path.replace("static/", "backend/static/")
            elif not green_box_path.startswith("backend/static/"):
                green_box_path = os.path.join("backend/static/results", os.path.basename(green_box_path))
            if not os.path.exists(green_box_path):
                # Fallback: search in backend/static/results
                alt_path = os.path.join("backend/static/results", os.path.basename(green_box_path))
                if os.path.exists(alt_path):
                    green_box_path = alt_path
            if os.path.exists(green_box_path):
                pdf.image(green_box_path, w=80)
            else:
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 7, f"Image not found: {green_box_path}", ln=True)
        except Exception as e:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 7, f"Image not found: {str(e)}", ln=True)
        # Unique visual
        if comp.get("visual_output"):
            try:
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 7, f"{comp.get('visual_label', '')}", ln=True)
                visual_path = comp["visual_output"]
                if visual_path.startswith("/static/"):
                    visual_path = visual_path.replace("/static/", "backend/static/")
                elif visual_path.startswith("static/"):
                    visual_path = visual_path.replace("static/", "backend/static/")
                elif not visual_path.startswith("backend/static/"):
                    visual_path = os.path.join("backend/static/results", os.path.basename(visual_path))
                if not os.path.exists(visual_path):
                    # Fallback: search in backend/static/results
                    alt_path = os.path.join("backend/static/results", os.path.basename(visual_path))
                    if os.path.exists(alt_path):
                        visual_path = alt_path
                if os.path.exists(visual_path):
                    pdf.image(visual_path, w=80)
                else:
                    pdf.set_font("Arial", 'I', 10)
                    pdf.cell(0, 7, f"Image not found: {visual_path}", ln=True)
            except Exception as e:
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 7, f"Image not found: {str(e)}", ln=True)
        pdf.ln(2)
    # Overall stats
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 10, f"Overall Similarity: {results['overall_similarity']}%", ln=True)
    pdf.ln(2)
    # Save PDF
    pdf_name = f"backend/static/results/visioniq_report_{os.path.basename(master_path)}.pdf"
    pdf.output(pdf_name)
    return pdf_name 