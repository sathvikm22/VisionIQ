from fpdf import FPDF
import os
from typing import List, Dict
from PIL import Image
import urllib.parse

def sanitize_filename(filename):
    safe_name = filename.replace(" ", "_").replace("%", "_").replace("&", "_")
    safe_name = safe_name.replace("+", "_").replace("#", "_").replace("?", "_")
    return urllib.parse.quote(safe_name)

def generate_pdf_report(master_path: str, processed_paths: List[str], results: Dict) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # 1. Heading
    pdf.set_font("Arial", 'B', 28)
    pdf.cell(0, 18, "Visual Analysis Report", ln=True, align='C')
    pdf.ln(2)
    # 2. Master image filename (label and filename aligned)
    pdf.set_font("Arial", 'B', 13)
    label = "Master Image filename: "
    label_width = pdf.get_string_width(label) + 2
    master_filename = os.path.basename(master_path)
    pdf.cell(label_width, 10, label, ln=0, align='L')
    x_filename = pdf.get_x()
    y_filename = pdf.get_y()
    pdf.multi_cell(0, 10, master_filename, align='L')
    pdf.ln(1)
    # 3. Master image preview
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
    # 4. Table header (with wrapped cells)
    pdf.set_font("Arial", 'B', 12)
    th = pdf.font_size + 3
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(30, 30, 30)
    col_widths = [80, 35, 35]  # Wider for filename
    headers = ["Filename", "Similarity %", "# Differences"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], th, h, border=1, align='C', fill=True)
    pdf.ln(th)
    pdf.set_font("Arial", '', 12)
    pdf.set_fill_color(255, 255, 255)
    for comp in results["comparisons"]:
        # Wrap filename in cell and track height
        x_left = pdf.get_x()
        y_top = pdf.get_y()
        pdf.set_xy(x_left, y_top)
        # Save current position for other cells
        pdf.multi_cell(col_widths[0], th, comp["filename"], border=1, align='C', fill=True)
        # Calculate height of the multi_cell
        y_bottom = pdf.get_y()
        cell_height = y_bottom - y_top
        # Move to the right for the next cell
        pdf.set_xy(x_left + col_widths[0], y_top)
        pdf.cell(col_widths[1], cell_height, str(comp["similarity_score"]), border=1, align='C', fill=True)
        pdf.cell(col_widths[2], cell_height, str(comp["num_differences"]), border=1, align='C', fill=True)
        pdf.ln(cell_height)
    pdf.ln(4)
    # Insert a page break before comparison images section
    pdf.add_page()
    # 5. Comparison images: filename, then two outputs side by side with headings
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 10, "Comparison Images with Differences Highlighted and Visuals:", ln=True)
    pdf.ln(2)
    comp_count = 0
    for comp in results["comparisons"]:
        if comp_count > 0 and comp_count % 2 == 0:
            pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(0, 8, comp['filename'], align='L')
        pdf.ln(1)
        # Prepare image paths
        green_box_path = comp["processed_image_url"]
        if green_box_path.startswith("/static/"):
            green_box_path = green_box_path.replace("/static/", "static/")
        elif green_box_path.startswith("static/"):
            green_box_path = green_box_path
        elif not green_box_path.startswith("static/"):
            green_box_path = os.path.join("static/results", os.path.basename(green_box_path))
        if not os.path.exists(green_box_path):
            from backend.utils.image_processing import create_placeholder_image
            placeholder_path = os.path.join("static/results", f"placeholder_{os.path.basename(green_box_path)}")
            create_placeholder_image(placeholder_path, "Image not found", (400, 400))
            green_box_path = placeholder_path
        visual_path = comp.get("visual_output")
        if visual_path:
            if visual_path.startswith("/static/"):
                visual_path = visual_path.replace("/static/", "static/")
            elif visual_path.startswith("static/"):
                visual_path = visual_path
            elif not visual_path.startswith("static/"):
                visual_path = os.path.join("static/results", os.path.basename(visual_path))
            if not os.path.exists(visual_path):
                from backend.utils.image_processing import create_placeholder_image
                placeholder_path = os.path.join("static/results", f"placeholder_{os.path.basename(visual_path)}")
                create_placeholder_image(placeholder_path, "Image not found", (400, 400))
                visual_path = placeholder_path
        # Headings for each image
        pdf.set_font("Arial", 'B', 10)
        y_start = pdf.get_y()
        x_start = pdf.get_x()
        pdf.cell(90, 8, "Differences Highlighted", border=0, align='C')
        pdf.cell(90, 8, comp.get('visual_label', ''), border=0, align='C')
        pdf.ln(8)
        # Images side by side
        y_img = pdf.get_y()
        x_img = pdf.get_x()
        def is_valid_image_path(path):
            return path and os.path.isfile(path) and path.lower().endswith((".png", ".jpg", ".jpeg"))
        if is_valid_image_path(green_box_path) and is_valid_image_path(visual_path):
            pdf.image(green_box_path, x=x_img, y=y_img, w=80)
            pdf.image(visual_path, x=x_img+90, y=y_img, w=80)
            pdf.ln(82)
        elif is_valid_image_path(green_box_path):
            pdf.image(green_box_path, w=80)
            pdf.ln(82)
        elif is_valid_image_path(visual_path):
            pdf.image(visual_path, w=80)
            pdf.ln(82)
        else:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 7, "Images not found", ln=True)
        pdf.ln(2)
        comp_count += 1
    # Save PDF
    pdf_base = f"visioniq_report_{os.path.basename(master_path)}.pdf"
    pdf_base = sanitize_filename(pdf_base)
    pdf_name = f"static/results/{pdf_base}"
    pdf.output(pdf_name)
    return pdf_name 