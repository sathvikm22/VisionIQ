import cv2
import numpy as np
import uuid
import logging
import os
from fastapi import UploadFile
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# Add imports for rembg and PIL for segmentation-based background removal
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
from PIL import Image as PILImage

# Upgrade CLIP model selection
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
    def get_feature_extractor():
        try:
            model, preprocess = clip.load("ViT-L/14")
        except Exception:
            model, preprocess = clip.load("ViT-B/32")
        model.eval()
        return model, preprocess
except ImportError:
    CLIP_AVAILABLE = False
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    def get_feature_extractor():
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
        if w > 2 and h > 2:  # Lowered from 5 to 2 to catch smaller differences
            cv2.rectangle(boxed, (x, y), (x+w, y+h), (0,255,0), 2)
    return boxed

def histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def resize_and_save(img, path, size=None):
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img)

# Improved background removal: try rembg, else fallback to contour
def remove_background(img):
    if REMBG_AVAILABLE:
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out = rembg_remove(pil_img)
        out_np = np.array(out)
        if out_np.shape[2] == 4:
            # Use alpha channel as mask
            mask = out_np[..., 3]
            result = cv2.bitwise_and(img, img, mask=mask)
            white_bg = np.ones_like(img) * 255
            result = np.where(mask[..., None] == 255, result, white_bg)
            return result
        else:
            return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    else:
        # Fallback: contour-based
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        largest = max(contours, key=cv2.contourArea)
        object_mask = np.zeros_like(gray)
        cv2.drawContours(object_mask, [largest], -1, 255, -1)
        result = cv2.bitwise_and(img, img, mask=object_mask)
        white_bg = np.ones_like(img) * 255
        result = np.where(object_mask[..., None] == 255, result, white_bg)
        return result

# Histogram normalization
def normalize_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[...,0] = cv2.equalizeHist(img_yuv[...,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Keypoint-based alignment (ORB + homography)
def align_images(img1, img2):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return img2  # fallback: no alignment
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return img2
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    if H is None:
        return img2
    aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    return aligned

def extract_features(img, model, preprocess):
    if 'clip' in globals() and CLIP_AVAILABLE:
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

async def analyze_images(master_path: str, comparison_images: list, algorithm: str, sensitivity: int):
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
        # Always use full resolution, only resize if shape mismatch
        if comp_gray.shape != master_gray.shape:
            comp_gray = cv2.resize(comp_gray, (master_gray.shape[1], master_gray.shape[0]))
            comp = cv2.resize(comp, (master_gray.shape[1], master_gray.shape[0]))
        visual_label = ""
        visual_output_path = ""
        processed_image_path = ""
        img_size = (400, 400)
        if algorithm == "SSIM":
            # Per-channel SSIM, then average
            ssim_scores = []
            diff_maps = []
            for c in range(3):
                score, diff = ssim(master[...,c], comp[...,c], full=True)
                ssim_scores.append(score)
                diff_maps.append(1 - diff)
            similarity = float(np.mean(ssim_scores)) * 100
            diff_map = np.mean(diff_maps, axis=0)
            threshold = max(0.01, np.mean(diff_map) + np.std(diff_map)/2)
            diff_mask = (diff_map > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            ssim_map = (diff_map * 255).astype(np.uint8)
            ssim_map_color = cv2.applyColorMap(ssim_map, cv2.COLORMAP_JET)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_ssim_heatmap_{comp_file.filename}.png"
            resize_and_save(ssim_map_color, visual_output_path, img_size)
            visual_label = "SSIM Heatmap"
        elif algorithm == "MSE":
            mse_scores = [mse(master[...,c], comp[...,c]) for c in range(3)]
            similarity = float(1/(1+np.mean(mse_scores))) * 100
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            error_map = cv2.applyColorMap(absdiff, cv2.COLORMAP_JET)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_mse_error_map_{comp_file.filename}.png"
            resize_and_save(error_map, visual_output_path, img_size)
            visual_label = "Pixel Error Map"
        elif algorithm == "PSNR":
            psnr_scores = [psnr(master[...,c], comp[...,c]) for c in range(3)]
            # Handle infinite or NaN PSNR (identical images)
            if np.any([np.isinf(s) or np.isnan(s) for s in psnr_scores]):
                similarity = 100.0
            else:
                similarity = float(np.mean(psnr_scores))
                similarity = min(max(similarity, 0), 100)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            noise_vis = cv2.applyColorMap(absdiff, cv2.COLORMAP_OCEAN)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_psnr_noise_{comp_file.filename}.png"
            resize_and_save(noise_vis, visual_output_path, img_size)
            visual_label = "Noise Difference Map"
        elif algorithm == "Histogram":
            scores = [histogram_similarity(master[...,c], comp[...,c]) for c in range(3)]
            similarity = float(np.mean(scores)) * 100
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
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
        elif algorithm == "Feature Matching":
            orb = cv2.ORB_create(2000)
            kp1, des1 = orb.detectAndCompute(master, None)
            kp2, des2 = orb.detectAndCompute(comp, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < np.mean([m.distance for m in matches]) + np.std([m.distance for m in matches])]
                similarity = float(len(good_matches) / max(len(kp1), len(kp2))) * 100 if max(len(kp1), len(kp2)) > 0 else 0
                match_img = cv2.drawMatches(master, kp1, comp, kp2, good_matches[:30], None, flags=2)
                visual_output_path = f"backend/static/results/{uuid.uuid4()}_feature_matches_{comp_file.filename}.png"
                resize_and_save(match_img, visual_output_path, img_size)
            else:
                similarity = 0
                visual_output_path = f"backend/static/results/{uuid.uuid4()}_feature_matches_{comp_file.filename}.png"
                create_placeholder_image(visual_output_path, "No keypoints found", img_size)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_label = "Keypoint Matches"
        elif algorithm == "Deep Learning":
            model, preprocess = get_feature_extractor()
            master_feat = extract_features(master, model, preprocess)
            comp_feat = extract_features(comp, model, preprocess)
            similarity = cosine_similarity(master_feat, comp_feat)
            similarity = max(min(similarity, 1.0), -1.0)
            similarity = (similarity + 1) / 2
            similarity = round(similarity * 100, 2)
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path = f"backend/static/results/{uuid.uuid4()}_green_boxes_{comp_file.filename}"
            resize_and_save(boxed, processed_image_path, img_size)
            processed_paths.append(processed_image_path)
            visual_output_path = f"backend/static/results/{uuid.uuid4()}_deep_diff_{comp_file.filename}.png"
            resize_and_save(absdiff, visual_output_path, img_size)
            visual_label = "Deep Cosine Similarity"
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
        similarities.append(similarity)
        results["comparisons"].append({
            "filename": comp_file.filename,
            "similarity_score": round(similarity, 2),
            "num_differences": int(num_diffs),
            "processed_image_url": processed_image_path.replace("backend/static", "static"),
            "visual_output": visual_output_path.replace("backend/static", "static") if visual_output_path else "",
            "visual_label": visual_label
        })
    results["overall_similarity"] = round(np.mean(similarities), 2) if similarities else 0
    return results, processed_paths 