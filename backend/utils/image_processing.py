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
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import logging

# Add logging setup
logging.basicConfig(level=logging.INFO)

# Try to use CLIP if available, else fallback to ResNet18
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18

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

def create_placeholder_image(path, label, size=(400, 400)):
    blank = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    cv2.putText(blank, label, (30, size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128,128,128), 2, cv2.LINE_AA)
    cv2.imwrite(path, blank)

def get_feature_extractor():
    if CLIP_AVAILABLE:
        model, preprocess = clip.load("ViT-B/32")
        model.eval()
        return model, preprocess
    else:
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform

def extract_features(img, model, preprocess):
    if CLIP_AVAILABLE:
        from PIL import Image as PILImage
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(img_tensor).cpu().numpy().squeeze()
        return features
    else:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
        return features

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def analyze_images(master_path: str, comparison_images: List[UploadFile], algorithm: str, sensitivity: int):
    master = cv2.imread(master_path)
    if master is None:
        raise ValueError(f"Failed to load master image from {master_path}. Is it a valid image file? Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
    master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    results = {"comparisons": []}
    similarities = []
    processed_paths = []
    for comp_file in comparison_images:
        comp_bytes = await comp_file.read()
        comp_arr = np.frombuffer(comp_bytes, np.uint8)
        comp = cv2.imdecode(comp_arr, cv2.IMREAD_COLOR)
        if comp is None:
            raise ValueError(f"Failed to load comparison image {comp_file.filename}. Is it a valid image file? Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        # Resize to match master
        if comp_gray.shape != master_gray.shape:
            comp_gray = cv2.resize(comp_gray, (master_gray.shape[1], master_gray.shape[0]))
            comp = cv2.resize(comp, (master_gray.shape[1], master_gray.shape[0]))
        visual_label = ""
        visual_output_path = ""
        processed_image_path = ""
        img_size = (400, 400)
        placeholder_needed = False
        if algorithm == "SSIM":
            score, diff = ssim(master, comp, full=True, channel_axis=2)
            similarity = float(score)
            diff_map = 1 - diff
            diff_mask = (diff_map > (sensitivity/200)).astype(np.uint8) * 255  # less strict
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            ssim_map = diff_map * 255
            ssim_map = ssim_map.astype(np.uint8)
            ssim_map_color = cv2.applyColorMap(ssim_map, cv2.COLORMAP_JET)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_ssim_heatmap_{comp_file.filename}.png"
            resize_and_save(ssim_map_color, visual_output_path, img_size)
            visual_label = "SSIM Heatmap"
            logging.info(f"Processed image path: {processed_image_path}")
            logging.info(f"Visual output path: {visual_output_path}")
        elif algorithm == "MSE":
            score = mse(master, comp)
            similarity = float(1/(1+score))
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (20 + 235 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            error_map = cv2.applyColorMap(absdiff, cv2.COLORMAP_JET)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_mse_error_map_{comp_file.filename}.png"
            resize_and_save(error_map, visual_output_path, img_size)
            visual_label = "Pixel Error Map"
            logging.info(f"Processed image path: {processed_image_path}")
            logging.info(f"Visual output path: {visual_output_path}")
        elif algorithm == "PSNR":
            score = psnr(master, comp)
            similarity = float(score/100)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (20 + 235 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            noise_vis = cv2.applyColorMap(absdiff, cv2.COLORMAP_OCEAN)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_psnr_noise_{comp_file.filename}.png"
            resize_and_save(noise_vis, visual_output_path, img_size)
            visual_label = "Noise Difference Map"
            logging.info(f"Processed image path: {processed_image_path}")
            logging.info(f"Visual output path: {visual_output_path}")
        elif algorithm == "Histogram":
            score = histogram_similarity(master, comp)
            similarity = float(score)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (20 + 235 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
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
            logging.info(f"Processed image path: {processed_image_path}")
            logging.info(f"Visual output path: {visual_output_path}")
        elif algorithm == "Feature Matching":
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(master, None)
            kp2, des2 = orb.detectAndCompute(comp, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                similarity = float(len(matches) / max(len(kp1), len(kp2))) if max(len(kp1), len(kp2)) > 0 else 0
                match_img = cv2.drawMatches(master, kp1, comp, kp2, matches[:30], None, flags=2)
                visual_output_path = f"backend/static/results/{uuid.uuid4()}_feature_matches_{comp_file.filename}.png"
                resize_and_save(match_img, visual_output_path, img_size)
            else:
                similarity = 0
                visual_output_path = f"backend/static/results/{uuid.uuid4()}_feature_matches_{comp_file.filename}.png"
                create_placeholder_image(visual_output_path, "No keypoints found", img_size)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            diff_mask = (diff_gray > (20 + 235 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_label = "Keypoint Matches"
            logging.info(f"Processed image path: {processed_image_path}")
            logging.info(f"Visual output path: {visual_output_path}")
        elif algorithm == "Deep Learning":
            model, preprocess = get_feature_extractor()
            master_feat = extract_features(master, model, preprocess)
            comp_feat = extract_features(comp, model, preprocess)
            similarity = cosine_similarity(master_feat, comp_feat)
            # Normalize similarity to 0-100% (CLIP/ResNet cosine sim is -1 to 1)
            similarity = (similarity + 1) / 2 * 100
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            # Use a lower threshold for difference mask
            diff_mask = (diff_gray > (30 + 225 * (sensitivity/100))).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_deep_diff_{comp_file.filename}.png"
            resize_and_save(absdiff, visual_output_path, img_size)
            visual_label = "Deep Cosine Similarity"
            logging.info(f"Processed image path: {processed_image_path}")
            logging.info(f"Visual output path: {visual_output_path}")
        else:
            similarity = 0
            diff_mask = np.zeros_like(master_gray)
            boxed = comp.copy()
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_visual_{comp_file.filename}.png"
            create_placeholder_image(visual_output_path, "No visual output", img_size)
            visual_label = ""
        # Count difference regions (contours)
        mask_for_contours = diff_mask
        if len(mask_for_contours.shape) == 3:
            mask_for_contours = cv2.cvtColor(mask_for_contours, cv2.COLOR_BGR2GRAY)
        if mask_for_contours.dtype != np.uint8:
            mask_for_contours = mask_for_contours.astype(np.uint8)
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_diffs = sum(1 for cnt in contours if cv2.contourArea(cnt) > 25)
        # Always ensure processed_image_path exists
        if not processed_image_path or not os.path.exists(processed_image_path):
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            create_placeholder_image(processed_image_path, "No differences found", img_size)
            processed_paths.append(processed_image_path)
        # Always ensure visual_output_path exists
        if not visual_output_path or not os.path.exists(visual_output_path):
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_visual_{comp_file.filename}.png"
            create_placeholder_image(visual_output_path, "No visual output", img_size)
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