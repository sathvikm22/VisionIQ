import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from fastapi import UploadFile
import os
import uuid
from typing import List, Tuple, Dict
from PIL import Image
import matplotlib.pyplot as plt

# Helper functions for each algorithm

def mse(img1, img2):
    return mean_squared_error(img1, img2)

def psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)

def histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def feature_matching(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None:
        return 0
    matches = bf.match(des1, des2)
    return len(matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0

def draw_bounding_boxes(image, diff_mask):
    # Ensure diff_mask is single-channel uint8
    if len(diff_mask.shape) == 3:
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
    if diff_mask.dtype != np.uint8:
        diff_mask = diff_mask.astype(np.uint8)
    # Find contours and draw green rectangles
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # filter small boxes
            cv2.rectangle(boxed, (x, y), (x+w, y+h), (0,255,0), 2)
    return boxed

def robust_diff_mask(diff, sensitivity):
    # Normalize diff to [0, 1] if not already
    if diff.max() > 1.0:
        diff = diff / 255.0
    # Sensitivity is 0-1, so threshold is 1-sensitivity
    thresh_val = 1 - sensitivity
    mask = (diff < thresh_val).astype(np.uint8) * 255
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

def resize_and_save(img, path, size=None):
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img)

async def analyze_images(master_path: str, comparison_images: List[UploadFile], algorithm: str, sensitivity: int):
    master = cv2.imread(master_path)
    master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    results = {"comparisons": []}
    similarities = []
    processed_paths = []
    for comp_file in comparison_images:
        comp_bytes = await comp_file.read()
        comp_arr = np.frombuffer(comp_bytes, np.uint8)
        comp = cv2.imdecode(comp_arr, cv2.IMREAD_COLOR)
        comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        # Resize to match master
        if comp_gray.shape != master_gray.shape:
            comp_gray = cv2.resize(comp_gray, (master_gray.shape[1], master_gray.shape[0]))
            comp = cv2.resize(comp, (master_gray.shape[1], master_gray.shape[0]))
        visual_label = ""
        visual_output_path = ""
        processed_image_path = ""
        img_size = (400, 400)
        if algorithm == "SSIM":
            score, diff = ssim(master, comp, full=True, channel_axis=2)
            similarity = float(score)
            # Use a robust threshold for SSIM: highlight regions where (1-diff) > sensitivity
            diff_map = 1 - diff
            diff_mask = (diff_map > (sensitivity/100)).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            # SSIM Heatmap (resize to 400x400)
            ssim_map = diff_map * 255
            ssim_map = ssim_map.astype(np.uint8)
            ssim_map_color = cv2.applyColorMap(ssim_map, cv2.COLORMAP_JET)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_ssim_heatmap_{comp_file.filename}.png"
            resize_and_save(ssim_map_color, visual_output_path, img_size)
            visual_label = "SSIM Heatmap"
        elif algorithm == "MSE":
            score = mse(master, comp)
            similarity = float(1/(1+score))
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (255 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            # Pixel Error Map (resize to 400x400)
            error_map = cv2.applyColorMap(absdiff, cv2.COLORMAP_JET)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_mse_error_map_{comp_file.filename}.png"
            resize_and_save(error_map, visual_output_path, img_size)
            visual_label = "Pixel Error Map"
        elif algorithm == "PSNR":
            score = psnr(master, comp)
            similarity = float(score/100)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (255 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            # Noise Difference Map (resize to 400x400)
            noise_vis = cv2.applyColorMap(absdiff, cv2.COLORMAP_OCEAN)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_psnr_noise_{comp_file.filename}.png"
            resize_and_save(noise_vis, visual_output_path, img_size)
            visual_label = "Noise Difference Map"
        elif algorithm == "Histogram":
            score = histogram_similarity(master, comp)
            similarity = float(score)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (255 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            # Color Histogram (resize to 400x400)
            plt.figure(figsize=(5, 4))
            for channel, color in zip(cv2.split(comp), ('b', 'g', 'r')):
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Color Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_histogram_{comp_file.filename}.png"
            plt.savefig(visual_output_path, dpi=120)
            plt.close()
            visual_label = "Color Histogram"
        elif algorithm == "Feature Matching":
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(master, None)
            kp2, des2 = orb.detectAndCompute(comp, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                similarity = float(len(matches) / max(len(kp1), len(kp2))) if max(len(kp1), len(kp2)) > 0 else 0
                match_img = cv2.drawMatches(master, kp1, comp, kp2, matches[:30], None, flags=2)
                visual_output_path = f"backend/static/results/{uuid.uuid4()}_feature_matches_{comp_file.filename}.png"
                resize_and_save(match_img, visual_output_path, img_size)
            else:
                similarity = 0
                visual_output_path = ""
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (255 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_label = "Keypoint Matches"
        else:
            similarity = 0
            diff_mask = np.zeros_like(master_gray)
            boxed = comp.copy()
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_output_path = ""
            visual_label = ""
        # Count difference regions (contours)
        mask_for_contours = diff_mask
        if len(mask_for_contours.shape) == 3:
            mask_for_contours = cv2.cvtColor(mask_for_contours, cv2.COLOR_BGR2GRAY)
        if mask_for_contours.dtype != np.uint8:
            mask_for_contours = mask_for_contours.astype(np.uint8)
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_diffs = sum(1 for cnt in contours if cv2.contourArea(cnt) > 25)
        similarities.append(similarity)
        results["comparisons"].append({
            "filename": comp_file.filename,
            "similarity_score": round(similarity*100, 2),
            "num_differences": int(num_diffs),
            "processed_image_url": processed_image_path.replace("backend/static", "static"),
            "visual_output": visual_output_path.replace("backend/static", "static") if visual_output_path else "",
            "visual_label": visual_label
        })
    results["overall_similarity"] = round(np.mean(similarities)*100, 2) if similarities else 0
    return results, processed_paths 