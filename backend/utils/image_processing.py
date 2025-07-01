import cv2
import numpy as np
import uuid
import logging
import os
from fastapi import UploadFile
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import insightface
from ultralytics import YOLO
import requests
import re
import urllib.parse

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

# DINOv2 and SAM imports
try:
    import torch
    import timm
    from segment_anything import sam_model_registry, SamPredictor
    import cv2
    import numpy as np
except ImportError:
    pass

# SigNet/ViT/SimCLR imports
try:
    from transformers import ViTModel, ViTFeatureExtractor
except ImportError:
    pass

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    pass

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
    if CLIP_AVAILABLE:
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

def get_insightface_model():
    # Download and cache ArcFace model if not present
    model_dir = os.path.expanduser("~/.insightface/models/arcface_r100_v1")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    try:
        model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(800, 800))  # Use larger detection window for robustness
        return model
    except Exception as e:
        return None

def get_yolov8_model():
    try:
        model = YOLO('yolov8n.pt')  # Use nano for speed, can be changed
        return model
    except Exception as e:
        return None

def get_dinov2_model():
    # DINOv2 ViT-L/14 weights (manual download may be required)
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dinov2_vitl14_pretrain.pth")
        if not os.path.exists(model_path):
            logging.warning(f"DINOv2 model not found at {model_path}")
            return None
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading DINOv2 model: {e}")
        return None

def get_sam_model():
    # SAM weights (manual download may be required)
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sam_vit_h_4b8939.pth")
        if not os.path.exists(model_path):
            logging.warning(f"SAM model not found at {model_path}")
            return None
        sam = sam_model_registry["vit_h"](model_path)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        logging.error(f"Error loading SAM model: {e}")
        return None

def get_signet_model():
    # TODO: Download and load SigNet weights, or provide instructions
    # For now, use ViT as a fallback
    try:
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        return model, extractor
    except Exception as e:
        return None, None

def face_verification(master, comp):
    model = get_insightface_model()
    if model is None:
        return 0.0, None, "InsightFace not available"
    
    try:
        # Multiple preprocessing attempts for better face detection
        preprocessing_attempts = [
            # Original RGB
            (cv2.cvtColor(master, cv2.COLOR_BGR2RGB), cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)),
            # Enhanced contrast
            (lambda: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2RGB),
             lambda: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2RGB)),
            # Gaussian blur for noise reduction
            (lambda: cv2.cvtColor(cv2.GaussianBlur(master, (3,3), 0), cv2.COLOR_BGR2RGB),
             lambda: cv2.cvtColor(cv2.GaussianBlur(comp, (3,3), 0), cv2.COLOR_BGR2RGB)),
            # Resized for better detection
            (lambda: cv2.cvtColor(cv2.resize(master, (512, 512)), cv2.COLOR_BGR2RGB),
             lambda: cv2.cvtColor(cv2.resize(comp, (512, 512)), cv2.COLOR_BGR2RGB))
        ]
        
        faces_master = None
        faces_comp = None
        master_rgb = None
        comp_rgb = None
        
        # Try different preprocessing methods
        for i, (master_proc, comp_proc) in enumerate(preprocessing_attempts):
            try:
                if callable(master_proc):
                    master_rgb = master_proc()
                    comp_rgb = comp_proc()
                else:
                    master_rgb = master_proc
                    comp_rgb = comp_proc
                
                faces_master = model.get(master_rgb)
                faces_comp = model.get(comp_rgb)
                
                if faces_master and faces_comp:
                    break
            except Exception as e:
                continue
        
        if not faces_master or not faces_comp:
            return 0.0, None, "No face detected in one or both images"
        
        # Get embeddings for all detected faces
        master_embeddings = [face.embedding for face in faces_master]
        comp_embeddings = [face.embedding for face in faces_comp]
        
        # Calculate similarity between all face pairs
        max_similarity = 0.0
        best_master_idx = 0
        best_comp_idx = 0
        
        for i, master_emb in enumerate(master_embeddings):
            for j, comp_emb in enumerate(comp_embeddings):
                similarity = cosine_similarity(master_emb, comp_emb)
                similarity = max(min(similarity, 1.0), -1.0)
                similarity = (similarity + 1) / 2 * 100
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_master_idx = i
                    best_comp_idx = j
        
        # Additional face analysis for better accuracy
        master_face = faces_master[best_master_idx]
        comp_face = faces_comp[best_comp_idx]
        
        # Get face attributes if available
        master_attributes = {}
        comp_attributes = {}
        
        if hasattr(master_face, 'age'):
            master_attributes['age'] = master_face.age
        if hasattr(comp_face, 'age'):
            comp_attributes['age'] = comp_face.age
            
        if hasattr(master_face, 'gender'):
            master_attributes['gender'] = master_face.gender
        if hasattr(comp_face, 'gender'):
            comp_attributes['gender'] = comp_face.gender
        
        # Adjust similarity based on attributes
        adjusted_similarity = max_similarity
        
        # Age difference penalty (if available)
        if 'age' in master_attributes and 'age' in comp_attributes:
            age_diff = abs(master_attributes['age'] - comp_attributes['age'])
            if age_diff > 10:  # Significant age difference
                adjusted_similarity *= 0.9
            elif age_diff > 5:  # Moderate age difference
                adjusted_similarity *= 0.95
        
        # Gender consistency check (if available)
        if 'gender' in master_attributes and 'gender' in comp_attributes:
            if master_attributes['gender'] != comp_attributes['gender']:
                adjusted_similarity *= 0.7  # Significant penalty for gender mismatch
        
        # Face quality assessment
        master_quality = get_face_quality(master_face)
        comp_quality = get_face_quality(comp_face)
        
        # Quality-based adjustment
        quality_factor = min(master_quality, comp_quality) / 100.0
        adjusted_similarity *= (0.8 + 0.2 * quality_factor)  # Quality affects 20% of score
        
        # Final similarity score
        final_similarity = min(adjusted_similarity, 100.0)
        
        # Create visualization
        vis = comp.copy()
        
        # Draw bounding boxes for all detected faces
        for i, f in enumerate(faces_comp):
            x1, y1, x2, y2 = map(int, f.bbox)
            color = (0, 255, 0) if i == best_comp_idx else (0, 0, 255)  # Green for best match, red for others
            thickness = 3 if i == best_comp_idx else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Add similarity score for the best match
            if i == best_comp_idx:
                cv2.putText(vis, f"Match: {final_similarity:.1f}%", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw keypoints if available
            if hasattr(f, 'kps') and f.kps is not None:
                for kp in f.kps:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(vis, (x, y), 2, (255,0,0), -1)
        
        # Add overall assessment text
        assessment = "SAME PERSON" if final_similarity > 70 else "DIFFERENT PERSON" if final_similarity < 40 else "UNCERTAIN"
        color = (0, 255, 0) if final_similarity > 70 else (0, 0, 255) if final_similarity < 40 else (0, 165, 255)
        
        cv2.putText(vis, f"Assessment: {assessment}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis, f"Confidence: {final_similarity:.1f}%", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return final_similarity, vis, None
        
    except Exception as e:
        logging.error(f"Face verification error: {e}")
        return 0.0, None, f"Face verification error: {str(e)}"

def get_face_quality(face):
    """Calculate face quality score based on various factors"""
    quality_score = 100.0
    
    # Check bounding box size (larger faces are generally better)
    bbox = face.bbox
    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if face_area < 1000:  # Very small face
        quality_score *= 0.7
    elif face_area < 5000:  # Small face
        quality_score *= 0.85
    
    # Check if keypoints are available and reasonable
    if hasattr(face, 'kps') and face.kps is not None:
        kps = face.kps
        # Check if keypoints are within reasonable bounds
        valid_kps = 0
        for kp in kps:
            if 0 <= kp[0] <= 1000 and 0 <= kp[1] <= 1000:  # Reasonable bounds
                valid_kps += 1
        
        if valid_kps < len(kps) * 0.8:  # Less than 80% valid keypoints
            quality_score *= 0.8
    
    # Check embedding norm (very small or large norms might indicate poor quality)
    if hasattr(face, 'embedding'):
        emb_norm = np.linalg.norm(face.embedding)
        if emb_norm < 0.1 or emb_norm > 10.0:
            quality_score *= 0.9
    
    return quality_score

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def machine_part_comparison(master, comp, orig_filename="machine_part.png"):
    """
    Robust machine part comparison using part-level (YOLO) matching, SSIM, ORB, Siamese, DINOv2, and pixel-wise difference.
    Part-level similarity and visualization are prioritized for accuracy.
    """
    import logging
    # --- 0. YOLOv8 Robust Part-Level Comparison ---
    yolo_score = None
    yolo_vis_path = None
    yolo_overlay = None
    matched, missing, extra = [], [], []
    try:
        model = get_yolov8_model()
        if model is not None:
            results_master = model(master, verbose=False)
            results_comp = model(comp, verbose=False)
            master_boxes = results_master[0].boxes.xyxy.cpu().numpy() if results_master[0].boxes is not None else []
            master_classes = results_master[0].boxes.cls.cpu().numpy().tolist() if results_master[0].boxes is not None else []
            comp_boxes = results_comp[0].boxes.xyxy.cpu().numpy() if results_comp[0].boxes is not None else []
            comp_classes = results_comp[0].boxes.cls.cpu().numpy().tolist() if results_comp[0].boxes is not None else []
            used_comp = set()
            # Match master to comp by class and IoU
            for i, (mbox, mcls) in enumerate(zip(master_boxes, master_classes)):
                best_iou = 0
                best_j = -1
                for j, (cbox, ccls) in enumerate(zip(comp_boxes, comp_classes)):
                    if j in used_comp: continue
                    if mcls != ccls: continue
                    iou = compute_iou(mbox, cbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou > 0.5 and best_j != -1:
                    matched.append((mbox, mcls, comp_boxes[best_j], comp_classes[best_j]))
                    used_comp.add(best_j)
                else:
                    missing.append((mbox, mcls))
            # Any comp boxes not matched are extra
            for j, (cbox, ccls) in enumerate(zip(comp_boxes, comp_classes)):
                if j not in used_comp:
                    extra.append((cbox, ccls))
            # Visualization
            vis = comp.copy()
            for mbox, mcls, cbox, ccls in matched:
                x1, y1, x2, y2 = map(int, cbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(vis, f"MATCH {model.names[int(ccls)]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            for mbox, mcls in missing:
                x1, y1, x2, y2 = map(int, mbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(vis, f"MISSING {model.names[int(mcls)]}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            for cbox, ccls in extra:
                x1, y1, x2, y2 = map(int, cbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(vis, f"EXTRA {model.names[int(ccls)]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            yolo_vis_path, _ = save_output_image(vis, f"{uuid.uuid4()}_yolo_parts", orig_filename)
            yolo_overlay = vis.copy()
            # Robust part-level similarity
            n_matched = len(matched)
            n_missing = len(missing)
            n_extra = len(extra)
            yolo_score = (2 * n_matched) / (2 * n_matched + n_missing + n_extra + 1e-6) * 100 if (n_matched + n_missing + n_extra) > 0 else 100.0
            logging.info(f"YOLO part-level: matched={n_matched}, missing={n_missing}, extra={n_extra}, score={yolo_score:.2f}")
        else:
            yolo_score = None
            yolo_vis_path = None
            yolo_overlay = None
            logging.warning("YOLOv8 model not available for part detection.")
    except Exception as e:
        yolo_score = None
        yolo_vis_path = None
        yolo_overlay = None
        logging.error(f"YOLO part detection error: {e}")

    # --- 1. SSIM (Structural Similarity Index) ---
    master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    ssim_score, ssim_diff = ssim(master_gray, comp_gray, full=True)
    ssim_diff_map = (1 - ssim_diff)
    ssim_threshold = max(0.005, np.mean(ssim_diff_map) + np.std(ssim_diff_map)/3)
    ssim_mask = (ssim_diff_map > ssim_threshold).astype(np.uint8) * 255
    ssim_mask = cv2.dilate(ssim_mask, np.ones((3,3), np.uint8), iterations=1)
    ssim_boxed = draw_bounding_boxes(comp, ssim_mask)
    ssim_boxed_path, _ = save_output_image(ssim_boxed, f"{uuid.uuid4()}_ssim_boxes", orig_filename)
    ssim_heatmap = cv2.applyColorMap((ssim_diff_map*255).astype(np.uint8), cv2.COLORMAP_JET)
    ssim_heatmap_path, _ = save_output_image(ssim_heatmap, f"{uuid.uuid4()}_ssim_heatmap", orig_filename)

    # --- 2. ORB Feature Matching ---
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(master_gray, None)
    kp2, des2 = orb.detectAndCompute(comp_gray, None)
    orb_score = 0
    orb_mask = np.zeros_like(master_gray)
    orb_vis_path = None
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < np.mean([m.distance for m in matches]) + 0.5*np.std([m.distance for m in matches])]
        orb_score = float(len(good_matches) / max(len(kp1), len(kp2))) if max(len(kp1), len(kp2)) > 0 else 0
        orb_vis = cv2.drawMatches(master, kp1, comp, kp2, good_matches[:30], None, flags=2)
        orb_vis_path, _ = save_output_image(orb_vis, f"{uuid.uuid4()}_orb_matches", orig_filename)
        matched_idx1 = set([m.queryIdx for m in good_matches])
        for i, kp in enumerate(kp1):
            if i not in matched_idx1:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(orb_mask, (x, y), 4, 255, -1)
    orb_mask = cv2.dilate(orb_mask, np.ones((3,3), np.uint8), iterations=1)

    # --- 3. Pixel-wise Absolute Difference ---
    absdiff = cv2.absdiff(master, comp)
    diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
    pix_threshold = max(5, np.mean(diff_gray) + 0.5*np.std(diff_gray))
    pix_mask = (diff_gray > pix_threshold).astype(np.uint8) * 255
    pix_mask = cv2.dilate(pix_mask, np.ones((3,3), np.uint8), iterations=1)
    absdiff_vis = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    absdiff_vis_path, _ = save_output_image(absdiff_vis, f"{uuid.uuid4()}_absdiff_heatmap", orig_filename)

    # --- 4. Combine All Masks for Final Error Map ---
    combined_mask = cv2.bitwise_or(ssim_mask, orb_mask)
    combined_mask = cv2.bitwise_or(combined_mask, pix_mask)
    combined_mask = cv2.dilate(combined_mask, np.ones((3,3), np.uint8), iterations=1)
    # Overlay YOLO boxes on the combined heatmap for unique right-side output
    combined_heatmap = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
    if yolo_overlay is not None:
        overlay = yolo_overlay.copy()
        alpha = 0.5
        cv2.addWeighted(combined_heatmap, alpha, overlay, 1-alpha, 0, overlay)
        unique_right_output = overlay
    else:
        unique_right_output = combined_heatmap
    combined_boxed = draw_bounding_boxes(comp, combined_mask)
    combined_boxed_path, _ = save_output_image(combined_boxed, f"{uuid.uuid4()}_combined_boxes", orig_filename)
    combined_heatmap_path, _ = save_output_image(combined_heatmap, f"{uuid.uuid4()}_combined_heatmap", orig_filename)
    unique_right_output_path, _ = save_output_image(unique_right_output, f"{uuid.uuid4()}_unique_right_output", orig_filename)

    # --- 5. Siamese Network (Deep Similarity) ---
    siamese_score = None
    siamese_vis_path = None
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        from torchvision.models import resnet18
        class SiameseNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = resnet18(pretrained=True)
                self.backbone.fc = nn.Identity()
            def forward(self, x1, x2):
                f1 = self.backbone(x1)
                f2 = self.backbone(x2)
                return f1, f2
        device = torch.device('cpu')
        model = SiameseNet().to(device).eval()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img1 = transform(master).unsqueeze(0)
        img2 = transform(comp).unsqueeze(0)
        with torch.no_grad():
            f1, f2 = model(img1, img2)
            siamese_score = float(torch.cosine_similarity(f1, f2).item())
            siamese_score = (siamese_score + 1) / 2 * 100
        siamese_vis = cv2.addWeighted(master, 0.5, comp, 0.5, 0)
        cv2.putText(siamese_vis, f"Siamese Similarity: {siamese_score:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        siamese_vis_path, _ = save_output_image(siamese_vis, f"{uuid.uuid4()}_siamese", orig_filename)
    except Exception as e:
        siamese_score = None
        siamese_vis_path = None

    # --- 6. DINOv2/ViT Feature Comparison ---
    dinov2_score = None
    dinov2_vis_path = None
    try:
        similarity, vis, _ = dinov2_sam_machine_part_comparison(master, comp)
        dinov2_score = similarity
        dinov2_vis_path, _ = save_output_image(vis, f"{uuid.uuid4()}_dinov2", orig_filename)
    except Exception as e:
        dinov2_score = None
        dinov2_vis_path = None

    # --- 7. Autoencoder-based Anomaly Detection (Optional) ---
    autoencoder_score = None
    autoencoder_heatmap_path = None
    try:
        # Placeholder: If you have an autoencoder, insert logic here
        pass
    except Exception as e:
        autoencoder_score = None
        autoencoder_heatmap_path = None

    # --- Aggregate Results ---
    results = {
        'yolo_score': yolo_score,
        'yolo_vis_path': yolo_vis_path,  # Left output: robust part-level boxes
        'unique_right_output_path': unique_right_output_path,  # Right output: unique combined visualization
        'ssim_score': float(ssim_score) * 100,
        'ssim_boxed_path': ssim_boxed_path,
        'ssim_heatmap_path': ssim_heatmap_path,
        'orb_score': float(orb_score) * 100,
        'orb_vis_path': orb_vis_path,
        'absdiff_vis_path': absdiff_vis_path,
        'combined_boxed_path': combined_boxed_path,
        'combined_heatmap_path': combined_heatmap_path,
        'siamese_score': siamese_score,
        'siamese_vis_path': siamese_vis_path,
        'dinov2_score': dinov2_score,
        'dinov2_vis_path': dinov2_vis_path,
        'autoencoder_score': autoencoder_score,
        'autoencoder_heatmap_path': autoencoder_heatmap_path,
    }
    # Use robust part-level similarity as main similarity
    main_similarity = yolo_score if yolo_score is not None else float(np.mean([v for v in [results['ssim_score'], results['orb_score'], results['siamese_score'], results['dinov2_score']] if v is not None]))
    main_vis = yolo_overlay if yolo_overlay is not None else comp
    return main_similarity, main_vis, results

def dinov2_sam_machine_part_comparison(master, comp):
    # Simplified version that doesn't take too long
    try:
        # Use DINOv2 for feature extraction only (faster)
        model = get_dinov2_model()
        if model is None:
            return 0.0, None, "DINOv2 not available"
        
        # Resize images for faster processing
        master_resized = cv2.resize(master, (224, 224))
        comp_resized = cv2.resize(comp, (224, 224))
        
        # Convert to RGB and normalize
        master_rgb = cv2.cvtColor(master_resized, cv2.COLOR_BGR2RGB) / 255.0
        comp_rgb = cv2.cvtColor(comp_resized, cv2.COLOR_BGR2RGB) / 255.0
        
        # Extract features (simplified)
        with torch.no_grad():
            master_tensor = torch.from_numpy(master_rgb).permute(2, 0, 1).unsqueeze(0).float()
            comp_tensor = torch.from_numpy(comp_rgb).permute(2, 0, 1).unsqueeze(0).float()
            
            # Use the model's forward method to get features
            master_output = model(master_tensor)
            comp_output = model(comp_tensor)
            
            # Extract features from the output
            if isinstance(master_output, dict):
                # If output is a dict, get the main features
                master_feat = master_output.get('x_norm_clstoken', master_output.get('x_norm_patchtokens', None))
                comp_feat = comp_output.get('x_norm_clstoken', comp_output.get('x_norm_patchtokens', None))
                
                if master_feat is None:
                    # Fallback to any available tensor
                    master_feat = list(master_output.values())[0]
                    comp_feat = list(comp_output.values())[0]
            else:
                # If output is a tensor directly
                master_feat = master_output
                comp_feat = comp_output
            
            # Ensure we have 1D feature vectors
            if len(master_feat.shape) > 1:
                master_feat = torch.mean(master_feat, dim=1).squeeze()
                comp_feat = torch.mean(comp_feat, dim=1).squeeze()
            
            # Calculate similarity
            similarity = cosine_similarity(master_feat.numpy(), comp_feat.numpy())
            similarity = max(min(similarity, 1.0), -1.0)
            similarity = (similarity + 1) / 2 * 100
        
        # Create visualization
        vis = comp.copy()
        cv2.putText(vis, f"DINOv2 Similarity: {similarity:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        return similarity, vis, None
        
    except Exception as e:
        logging.error(f"DINOv2+SAM comparison error: {e}")
        return 0.0, None, f"DINOv2+SAM comparison error: {str(e)}"

def detect_signature_or_seal_regions(img):
    # --- Fast classical method first ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    thresh1 = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    _, thresh2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=2)
    regions = []
    h, w = img.shape[:2]
    img_area = h * w
    for mask in [morph1, morph2]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.003 * img_area or area > 0.99 * img_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(ch)
            is_signature = 1.2 < aspect < 15 or cw > 0.7 * w or ch > 0.7 * h
            is_seal = 0.5 < aspect < 2.0 and (4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-5)) > 0.5
            if is_signature or is_seal:
                regions.append((x, y, cw, ch, area, 'signature' if is_signature else 'seal'))
    if regions:
        return regions
    # --- Only use SAM if classical method fails ---
    try:
        predictor = get_sam_model()
        if predictor is not None:
            h, w = img.shape[:2]
            scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
            if scale < 1.0:
                img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
            else:
                img_small = img
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_rgb)
            masks = predictor.predict(multi_mask_output=True)
            regions = []
            h_s, w_s = img_small.shape[:2]
            img_area_s = h_s * w_s
            for mask in masks:
                m = mask["segmentation"] if isinstance(mask, dict) else mask
                m_uint8 = (m * 255).astype(np.uint8)
                contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 0.003 * img_area_s or area > 0.99 * img_area_s:
                        continue
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if scale < 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        cw = int(cw / scale)
                        ch = int(ch / scale)
                    aspect = cw / float(ch)
                    is_signature = 1.2 < aspect < 15 or cw > 0.7 * w or ch > 0.7 * h
                    is_seal = 0.5 < aspect < 2.0 and (4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-5)) > 0.5
                    if is_signature or is_seal:
                        regions.append((x, y, cw, ch, area, 'signature' if is_signature else 'seal'))
            if regions:
                return regions
    except Exception as e:
        logging.error(f"SAM signature detection error: {e}")
    # Fallback: if no region found, but the image is not blank, use full image
    if not regions and np.count_nonzero(morph1) > 0.01 * img_area:
        regions.append((0, 0, w, h, img_area, 'signature'))
    return regions

def auto_rotate_to_portrait(img):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if w > h:
        # Rotate 90 degrees to make portrait
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def signature_verification_sota(master, comp):
    model, extractor = get_signet_model()
    if model is None or extractor is None:
        return signature_verification(master, comp)
    try:
        def get_signature_crop(img):
            regions = detect_signature_or_seal_regions(img)
            if not regions:
                return None, None
            sig_regions = [r for r in regions if r[-1] == 'signature']
            if not sig_regions:
                return None, None
            x, y, w, h, *_ = max(sig_regions, key=lambda r: r[2]*r[3])
            crop = img[y:y+h, x:x+w]
            return crop, (x, y, w, h)
        master_crop, master_bbox = get_signature_crop(master)
        comp_crop, comp_bbox = get_signature_crop(comp)
        if master_crop is None or comp_crop is None or master_crop.size == 0 or comp_crop.size == 0:
            composite = np.ones((180, 360, 3), dtype=np.uint8) * 255
            cv2.putText(composite, "No signature detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            from .image_processing import save_output_image
            save_output_image(composite, f"{uuid.uuid4()}_signature", "signature_output.png")
            save_output_image(composite, f"{uuid.uuid4()}_green_boxes", "signature_output.png")
            return 0.0, composite, None
        # --- Auto-rotate to portrait orientation ---
        master_crop = auto_rotate_to_portrait(master_crop)
        comp_crop = auto_rotate_to_portrait(comp_crop)
        # --- Alignment ---
        comp_crop_aligned = align_images(master_crop, comp_crop)
        if comp_crop_aligned is None or comp_crop_aligned.size == 0 or comp_crop_aligned.shape != master_crop.shape:
            comp_crop_aligned = comp_crop.copy()
        crop_h, crop_w = master_crop.shape[:2]
        try:
            comp_crop_aligned = cv2.resize(comp_crop_aligned, (crop_w, crop_h))
        except Exception:
            comp_crop_aligned = comp_crop.copy()
            comp_crop_aligned = cv2.resize(comp_crop_aligned, (crop_w, crop_h))
        # --- Similarity calculation (unchanged) ---
        from PIL import Image
        master_pil = Image.fromarray(cv2.cvtColor(master_crop, cv2.COLOR_BGR2RGB))
        comp_pil = Image.fromarray(cv2.cvtColor(comp_crop_aligned, cv2.COLOR_BGR2RGB))
        master_inputs = extractor(images=master_pil, return_tensors="pt")
        comp_inputs = extractor(images=comp_pil, return_tensors="pt")
        with torch.no_grad():
            master_emb = model(**master_inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            comp_emb = model(**comp_inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        similarity = cosine_similarity(master_emb, comp_emb)
        similarity = max(min(similarity, 1.0), -1.0)
        similarity = (similarity + 1) / 2 * 100
        # --- Finer green box detection using Canny edges ---
        master_gray = cv2.cvtColor(master_crop, cv2.COLOR_BGR2GRAY)
        comp_gray = cv2.cvtColor(comp_crop_aligned, cv2.COLOR_BGR2GRAY)
        absdiff = cv2.absdiff(master_gray, comp_gray)
        edges = cv2.Canny(absdiff, 30, 100)
        kernel = np.ones((1,1), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxed = np.ones_like(comp_crop_aligned) * 255
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0.2:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 0.7*crop_w and h < 0.7*crop_h:
                    cv2.rectangle(boxed, (x, y), (x+w, y+h), (0,255,0), 1)
        # --- Compose output visualization ---
        vis = np.ones((crop_h, crop_w*2, 3), dtype=np.uint8) * 255
        vis[:, :crop_w] = master_crop
        vis[:, crop_w:] = comp_crop_aligned
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = f"ViT Similarity: {similarity:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        margin = 5
        cv2.rectangle(vis, (margin, margin), (margin + text_w + 4, margin + text_h + 8), (255,255,255), -1)
        cv2.putText(vis, text, (margin + 2, margin + text_h + 2), font, font_scale, (0,200,0), thickness, cv2.LINE_AA)
        from .image_processing import save_output_image, sanitize_filename
        base_name = sanitize_filename("signature_output.png")
        save_output_image(vis, f"{uuid.uuid4()}_signature", base_name)
        save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", base_name)
        return similarity, vis, None
    except Exception as e:
        logging.error(f"ViT signature verification error: {e}")
        composite = np.ones((180, 360, 3), dtype=np.uint8) * 255
        cv2.putText(composite, "Signature verification failed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        from .image_processing import save_output_image
        save_output_image(composite, f"{uuid.uuid4()}_signature", "signature_output.png")
        save_output_image(composite, f"{uuid.uuid4()}_green_boxes", "signature_output.png")
        return 0.0, composite, None

def signature_verification(master, comp):
    # Always use the ViT/ResNet fallback for signature verification
    return signature_verification_sota(master, comp)

def create_placeholder_image(path, text, size=(400, 400)):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', size, color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    font_size = 48 if size[0] >= 400 else 24
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(0,0)]:
        d.text((x+dx, y+dy), text, fill=(30, 30, 30), font=font)
    img.save(path)

# --- Centralized output image saving and URL generation ---
def sanitize_filename(filename):
    safe_name = filename.replace(" ", "_").replace("%", "_").replace("&", "_")
    safe_name = safe_name.replace("+", "_").replace("#", "_").replace("?", "_")
    return urllib.parse.quote(safe_name)

def make_output_path(prefix, orig_filename, ext=None):
    base = f"{prefix}_{sanitize_filename(orig_filename)}"
    if ext and not base.endswith(ext):
        base += ext
    return f"static/results/{base}"

def save_output_image(img, prefix, orig_filename, img_size=(400, 400), ext='.png'):
    import logging
    # Sanitize and limit filename
    safe_name = sanitize_filename(orig_filename)
    if len(safe_name) > 60:
        safe_name = safe_name[-60:]
    base = f"{prefix}_{safe_name}"
    if not base.endswith('.png'):
        base += '.png'
    path = f"static/results/{base}"
    try:
        if isinstance(img, str) and img.startswith('PLACEHOLDER:'):
            create_placeholder_image(path, img[12:], img_size)
            logging.info(f"Created placeholder image: {path}")
        else:
            if img is not None:
                if img.shape[0] != img_size[1] or img.shape[1] != img_size[0]:
                    img = cv2.resize(img, img_size)
                success = cv2.imwrite(path, img)
                if not success:
                    logging.error(f"cv2.imwrite failed for {path}, creating placeholder.")
                    create_placeholder_image(path, "Image not found", img_size)
                else:
                    logging.info(f"Saved image: {path}")
            else:
                create_placeholder_image(path, "Image not found", img_size)
                logging.warning(f"Input image was None, created placeholder: {path}")
        # Check if file exists
        import os
        if not os.path.exists(path):
            logging.error(f"Image file does not exist after save: {path}")
    except Exception as e:
        logging.error(f"Exception saving image {path}: {e}")
        create_placeholder_image(path, "Image not found", img_size)
    return path, url_with_leading_slash(path)

def detect_largest_face_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None  # No face found
    # Find largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    cropped = img[y:y+h, x:x+w]
    return cropped

def is_plausible_face(face, img_shape, min_area_ratio=0.02, max_aspect_ratio=2.0):
    # Check bounding box area and aspect ratio
    x1, y1, x2, y2 = map(int, face.bbox)
    area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    aspect_ratio = max((x2 - x1) / max(1, (y2 - y1)), (y2 - y1) / max(1, (x2 - x1)))
    if area < min_area_ratio * img_area:
        return False
    if aspect_ratio > max_aspect_ratio:
        return False
    # If face has detection score, require >0.5
    if hasattr(face, 'det_score') and face.det_score is not None:
        if face.det_score < 0.5:
            return False
    return True

async def analyze_images(master_path: str, comparison_images: list, algorithm: str, sensitivity: int):
    master = cv2.imread(master_path)
    if master is None or master.size == 0:
        raise ValueError(f"Failed to load master image from {master_path}. The file may be missing, corrupted, or not an image.")
    master = normalize_image(master)
    master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    results = {"comparisons": []}
    similarities = []
    processed_paths = []
    
    for comp_file in comparison_images:
        comp_bytes = await comp_file.read()
        comp_arr = np.frombuffer(comp_bytes, np.uint8)
        comp = cv2.imdecode(comp_arr, cv2.IMREAD_COLOR)
        if comp is None or comp.size == 0:
            raise ValueError(f"Failed to load comparison image: {getattr(comp_file, 'filename', 'unknown')}. The file may be missing, corrupted, or not an image.")
        comp = normalize_image(comp)
        comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        # Always use full resolution, only resize if shape mismatch
        if comp_gray.shape != master_gray.shape:
            comp_gray = cv2.resize(comp_gray, (master_gray.shape[1], master_gray.shape[0]))
            comp = cv2.resize(comp, (master_gray.shape[1], master_gray.shape[0]))
        visual_label = ""
        visual_output_path = ""
        processed_image_path = ""
        img_size = (400, 400)
        
        # Initialize diff_mask to prevent UnboundLocalError
        diff_mask = np.zeros_like(master_gray)
        
        # Initialize num_diffs to prevent UnboundLocalError
        num_diffs = 0
        
        # --- Face Verification: Crop faces for document images ---
        if algorithm == "Face Verification":
            model = get_insightface_model()
            if model is None:
                similarity = 0.0
                assessment = "Face detection model not available"
                crop_size = 180
                master_face_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 240
                comp_face_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 240
                composite = np.ones((crop_size, crop_size*2, 3), dtype=np.uint8) * 255
                composite[:, :crop_size] = master_face_crop
                composite[:, crop_size:] = comp_face_crop
                border_color = (0,0,255)
                cv2.rectangle(composite, (0,0), (crop_size-1, crop_size-1), border_color, 3)
                cv2.rectangle(composite, (crop_size,0), (crop_size*2-1, crop_size-1), border_color, 3)
                cv2.putText(composite, f"Similarity: {similarity:.1f}%", (10, crop_size-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 3)
                cv2.putText(composite, f"{assessment}", (10, crop_size-45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 3)
                composite_output_path, composite_url = save_output_image(composite, f"{uuid.uuid4()}_face_crops", "model_not_available.png")
                vis = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 240
                cv2.putText(vis, f"Assessment: {assessment}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
                cv2.putText(vis, f"Confidence: {similarity:.1f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
                processed_image_path, processed_url = save_output_image(vis, f"{uuid.uuid4()}_green_boxes", "model_not_available.png")
                processed_paths.append(processed_image_path)
                results["comparisons"].append({
                    "filename": "model_not_available.png",
                    "similarity_score": similarity,
                    "num_differences": 0,
                    "processed_image_url": url_with_leading_slash(processed_image_path),
                    "visual_output": url_with_leading_slash(composite_output_path),
                    "visual_label": "Face Crop Comparison",
                    "assessment": assessment
                })
                continue
            def is_valid_face(face):
                # Only accept faces with high confidence and at least 5 keypoints
                return (hasattr(face, 'det_score') and face.det_score is not None and face.det_score >= 0.7 and
                        hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5)
            def pad_image(img, pad_ratio=0.3, color=(255,255,255)):
                h, w = img.shape[:2]
                pad_h = int(h * pad_ratio)
                pad_w = int(w * pad_ratio)
                return cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=color)
            def multiscale_face_detect(img, model, scales=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1], pad_attempts=[0, 0.3, 0.5]):
                h, w = img.shape[:2]
                best_face = None
                best_scale = 1.0
                best_area = 0
                best_faces = None
                for pad in pad_attempts:
                    if pad > 0:
                        padded_img = pad_image(img, pad)
                    else:
                        padded_img = img
                    ph, pw = padded_img.shape[:2]
                    for scale in scales:
                        if scale == 1.0:
                            scaled_img = padded_img
                        else:
                            scaled_img = cv2.resize(padded_img, (int(pw*scale), int(ph*scale)))
                        faces = model.get(cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB))
                        # Filter only valid faces
                        faces = [f for f in faces if is_valid_face(f)] if faces else []
                        if faces:
                            # Pick the largest valid face at this scale
                            f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                            area = (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
                            if area > best_area:
                                best_face = f
                                best_scale = scale
                                best_area = area
                                best_faces = faces
                    if best_face is not None:
                        # Map bbox and keypoints back to original image coordinates if padded or scaled
                        scale_factor = best_scale
                        pad_h = int(h * pad)
                        pad_w = int(w * pad)
                        def unpad_and_scale_bbox(bbox, scale, pad_w, pad_h):
                            return [max(0, int(b/scale - pad_w if i%2==0 else b/scale - pad_h)) for i, b in enumerate(bbox)]
                        best_face.bbox = unpad_and_scale_bbox(best_face.bbox, scale_factor, pad_w, pad_h)
                        if hasattr(best_face, 'kps') and best_face.kps is not None:
                            best_face.kps = [[kp[0]/scale_factor - pad_w, kp[1]/scale_factor - pad_h] for kp in best_face.kps]
                        # Also map all faces for multi-face UI
                        if best_faces:
                            for f in best_faces:
                                f.bbox = unpad_and_scale_bbox(f.bbox, scale_factor, pad_w, pad_h)
                                if hasattr(f, 'kps') and f.kps is not None:
                                    f.kps = [[kp[0]/scale_factor - pad_w, kp[1]/scale_factor - pad_h] for kp in f.kps]
                        return best_face, best_faces
                return None, None
            # Use multi-scale, padded detection for both master and comparison images
            master_face, faces_master = multiscale_face_detect(master, model)
            comp_face, faces_comp = multiscale_face_detect(comp, model)
            # If no valid face in master or comparison image, only show right-side message
            if not master_face or not faces_comp:
                similarity = 0.0
                assessment = "No human face detected in one or both images."
                crop_size = 180
                master_face_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 240
                comp_face_crop = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 240
                composite = np.ones((crop_size, crop_size*2, 3), dtype=np.uint8) * 255
                composite[:, :crop_size] = master_face_crop
                composite[:, crop_size:] = comp_face_crop
                border_color = (0,0,255)
                cv2.rectangle(composite, (0,0), (crop_size-1, crop_size-1), border_color, 3)
                cv2.rectangle(composite, (crop_size,0), (crop_size*2-1, crop_size-1), border_color, 3)
                cv2.putText(composite, f"Similarity: {similarity:.1f}%", (10, crop_size-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 3)
                cv2.putText(composite, f"{assessment}", (10, crop_size-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
                composite_output_path, composite_url = save_output_image(composite, f"{uuid.uuid4()}_face_crops", comp_file.filename)
                processed_image_path = None
                results["comparisons"].append({
                    "filename": comp_file.filename,
                    "similarity_score": similarity,
                    "num_differences": 0,
                    "processed_image_url": "",  # No left box
                    "visual_output": url_with_leading_slash(composite_output_path),
                    "visual_label": "Face Crop Comparison",
                    "assessment": assessment
                })
                continue
            # For each face in comparison image, compare to master
            similarities = []
            for f in faces_comp:
                sim = cosine_similarity(master_face.embedding, f.embedding)
                sim = max(min(sim, 1.0), -1.0)
                sim = (sim + 1) / 2 * 100
                similarities.append(sim)
            best_idx = int(np.argmax(similarities))
            best_similarity = similarities[best_idx]
            def align_and_crop(img, face):
                if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
                    left_eye, right_eye = face.kps[0], face.kps[1]
                    dx = right_eye[0] - left_eye[0]
                    dy = right_eye[1] - left_eye[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
                    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    x1, y1, x2, y2 = map(int, face.bbox)
                    crop = aligned[y1:y2, x1:x2]
                    return crop
                else:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    return img[y1:y2, x1:x2]
            crop_size = 180
            def safe_resize_or_blank(crop, size=(180,180)):
                if crop is None or crop.size == 0 or len(crop.shape) != 3 or crop.shape[2] != 3:
                    return np.ones((size[0], size[1], 3), dtype=np.uint8) * 240
                if crop.shape[0] != size[0] or crop.shape[1] != size[1]:
                    return cv2.resize(crop, size)
                return crop
            master_face_crop = safe_resize_or_blank(align_and_crop(master, master_face), (crop_size, crop_size))
            comp_face_crop = safe_resize_or_blank(align_and_crop(comp, faces_comp[best_idx]), (crop_size, crop_size))
            composite = np.ones((crop_size, crop_size*2, 3), dtype=np.uint8) * 255
            composite[:, :crop_size] = master_face_crop
            composite[:, crop_size:] = comp_face_crop
            border_color = (0,255,0) if best_similarity > 70 else (0,0,255)
            assessment = "Same Person" if best_similarity > 70 else "Different Person"
            cv2.rectangle(composite, (0,0), (crop_size-1, crop_size-1), border_color, 3)
            cv2.rectangle(composite, (crop_size,0), (crop_size*2-1, crop_size-1), border_color, 3)
            cv2.putText(composite, f"Similarity: {best_similarity:.1f}%", (10, crop_size-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 3)
            cv2.putText(composite, f"{assessment}", (10, crop_size-45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 3)
            composite_output_path, composite_url = save_output_image(composite, f"{uuid.uuid4()}_face_crops", comp_file.filename)
            vis = comp.copy()
            for i, f in enumerate(faces_comp):
                x1, y1, x2, y2 = map(int, f.bbox)
                color = (0,255,0) if i == best_idx and best_similarity > 70 else (0,0,255)
                thickness = 3 if i == best_idx and best_similarity > 70 else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                if hasattr(f, 'kps') and f.kps is not None:
                    for kp in f.kps:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(vis, (x, y), 2, (255,0,0), -1)
            cv2.putText(vis, f"Assessment: {assessment}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
            cv2.putText(vis, f"Confidence: {best_similarity:.1f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
            processed_image_path, processed_url = save_output_image(vis, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            results["comparisons"].append({
                "filename": comp_file.filename,
                "similarity_score": round(best_similarity, 2),
                "num_differences": 0,
                "processed_image_url": url_with_leading_slash(processed_image_path),
                "visual_output": url_with_leading_slash(composite_output_path),
                "visual_label": "Face Crop Comparison",
                "assessment": assessment
            })
            continue
        if algorithm == "SSIM":
            ssim_scores = []
            diff_maps = []
            for c in range(3):
                score, diff = ssim(master[...,c], comp[...,c], full=True)
                ssim_scores.append(score)
                diff_maps.append(1 - diff)
            similarity = float(np.mean(ssim_scores)) * 100
            diff_map = np.mean(diff_maps, axis=0)
            threshold = max(0.01, np.mean(diff_map) + np.std(diff_map)/2)
            diff_mask = (diff_map > threshold).astype(np.uint8)
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            ssim_map = (diff_map * 255).astype(np.uint8)
            ssim_map_color = cv2.applyColorMap(ssim_map, cv2.COLORMAP_JET)
            visual_output_path, visual_url = save_output_image(ssim_map_color, f"{uuid.uuid4()}_ssim_heatmap", comp_file.filename)
            visual_label = "SSIM Heatmap"
        elif algorithm == "MSE":
            mse_scores = [mse(master[...,c], comp[...,c]) for c in range(3)]
            mse_value = np.mean(mse_scores)
            similarity = max(0, min(100, 100 - mse_value / (mse_value + 1) * 100))
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            error_map = cv2.applyColorMap(absdiff, cv2.COLORMAP_JET)
            visual_output_path, visual_url = save_output_image(error_map, f"{uuid.uuid4()}_mse_error_map", comp_file.filename)
            visual_label = "Pixel Error Map"
        elif algorithm == "PSNR":
            psnr_scores = [psnr(master[...,c], comp[...,c]) for c in range(3)]
            if np.any([np.isinf(s) or np.isnan(s) for s in psnr_scores]):
                similarity = 100.0
            else:
                psnr_value = np.mean(psnr_scores)
                similarity = max(0, min(100, (psnr_value / 50.0) * 100))
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            noise_vis = cv2.applyColorMap(absdiff, cv2.COLORMAP_OCEAN)
            visual_output_path, visual_url = save_output_image(noise_vis, f"{uuid.uuid4()}_psnr_noise", comp_file.filename)
            visual_label = "Noise Difference Map"
        elif algorithm == "Histogram":
            scores = [histogram_similarity(master[...,c], comp[...,c]) for c in range(3)]
            similarity = float(np.mean(scores)) * 100
            absdiff = cv2.absdiff(master, comp)
            diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
            threshold = np.mean(diff_gray) + np.std(diff_gray)
            diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
            boxed = draw_bounding_boxes(comp, diff_mask)
            processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            import io
            buf = io.BytesIO()
            plt.figure(figsize=(5, 4))
            for channel, color in zip(cv2.split(comp), ('b', 'g', 'r')):
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Color Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            hist_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            hist_img = cv2.imdecode(hist_img, cv2.IMREAD_COLOR)
            visual_output_path, visual_url = save_output_image(hist_img, f"{uuid.uuid4()}_histogram", comp_file.filename)
            visual_label = "Color Histogram"
        elif algorithm == "Feature Matching":
            if np.array_equal(master, comp):
                similarity = 100.0
                num_diffs = 0
                orb = cv2.ORB_create(2000)
                kp1, des1 = orb.detectAndCompute(master, None)
                kp2, des2 = orb.detectAndCompute(comp, None)
                match_img = None
                if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    match_img = cv2.drawMatches(master, kp1, comp, kp2, matches[:30], None, flags=2)
                if match_img is not None:
                    visual_output_path, visual_url = save_output_image(match_img, f"{uuid.uuid4()}_feature_matches", comp_file.filename)
                else:
                    visual_output_path, visual_url = save_output_image('PLACEHOLDER:No keypoints found', f"{uuid.uuid4()}_feature_matches", comp_file.filename)
                boxed = comp.copy()
                # Only draw green boxes if there are real differences
                diff = cv2.absdiff(master, comp)
                if np.max(diff) > 10:  # threshold for real difference
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    threshold = np.mean(diff_gray) + np.std(diff_gray)
                    diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
                    boxed = draw_bounding_boxes(comp, diff_mask)
                processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
                processed_paths.append(processed_image_path)
                visual_label = "Keypoint Matches (Identical Images)"
            else:
                orb = cv2.ORB_create(2000)
                kp1, des1 = orb.detectAndCompute(master, None)
                kp2, des2 = orb.detectAndCompute(comp, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = [m for m in matches if m.distance < np.mean([m.distance for m in matches]) + np.std([m.distance for m in matches])]
                    num_diffs = 0 if len(good_matches) == max(len(kp1), len(kp2)) else 1
                    similarity = float(len(good_matches) / max(len(kp1), len(kp2))) * 100 if max(len(kp1), len(kp2)) > 0 else 0
                    if num_diffs == 0:
                        similarity = 100.0
                    match_img = cv2.drawMatches(master, kp1, comp, kp2, good_matches[:30], None, flags=2)
                    visual_output_path, visual_url = save_output_image(match_img, f"{uuid.uuid4()}_feature_matches", comp_file.filename)
                else:
                    similarity = 0
                    num_diffs = 1
                    visual_output_path, visual_url = save_output_image('PLACEHOLDER:No keypoints found', f"{uuid.uuid4()}_feature_matches", comp_file.filename)
                absdiff = cv2.absdiff(master, comp)
                diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
                threshold = np.mean(diff_gray) + np.std(diff_gray)
                diff_mask = (diff_gray > threshold).astype(np.uint8) * 255
                boxed = draw_bounding_boxes(comp, diff_mask)
                processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
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
            processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            visual_output_path, visual_url = save_output_image(absdiff, f"{uuid.uuid4()}_deep_diff", comp_file.filename)
            visual_label = "Deep Cosine Similarity"
        elif algorithm == "Machine Part Comparison":
            try:
                similarity, vis, err = machine_part_comparison(master, comp)
                if err:
                    raise Exception(err)
            except Exception as e:
                # Fallback to SSIM
                ssim_score = ssim(master_gray, comp_gray)
                similarity = float(ssim_score) * 100
                vis = draw_bounding_boxes(comp, (cv2.absdiff(master, comp).mean(axis=2) > 20).astype(np.uint8) * 255)
                visual_label = "Fallback: SSIM"
            visual_output_path, visual_url = save_output_image(vis, f"{uuid.uuid4()}_machine_part", comp_file.filename)
            processed_image_path, processed_url = save_output_image(vis, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
        elif algorithm == "Signature Verification":
            similarity, vis, err = signature_verification_sota(master, comp)
            if err and "not available" in err:
                similarity, vis, err = signature_verification_sota(master, comp)
                visual_label = "Signature Verification (ViT/SigNet)"
            else:
                visual_label = "Signature Verification (ViT/SigNet)"
            if vis is not None:
                visual_output_path, visual_url = save_output_image(vis, f"{uuid.uuid4()}_signature", comp_file.filename)
            else:
                visual_output_path, visual_url = save_output_image('PLACEHOLDER:Signature Verification Error', f"{uuid.uuid4()}_signature", comp_file.filename)
            # Draw green boxes for all detected regions in comp image
            if vis is not None and hasattr(vis, '_comp_regions'):
                diff_img = comp.copy()
                for rx, ry, rw, rh, _, _ in vis._comp_regions:
                    cv2.rectangle(diff_img, (rx, ry), (rx+rw, ry+rh), (0,255,0), 3)
                processed_image_path, processed_url = save_output_image(diff_img, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            else:
                diff_mask = cv2.absdiff(master, comp).mean(axis=2) > 20
                diff_mask = diff_mask.astype(np.uint8) * 255
                boxed = draw_bounding_boxes(comp, diff_mask)
                processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
        else:
            similarity = 0
            diff_mask = np.zeros_like(master_gray)
            boxed = comp.copy()
            processed_image_path, processed_url = save_output_image(boxed, f"{uuid.uuid4()}_green_boxes", comp_file.filename)
            processed_paths.append(processed_image_path)
            visual_output_path, visual_url = save_output_image('PLACEHOLDER:No visual output', f"{uuid.uuid4()}_visual", comp_file.filename)
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
        # At the end of each comparison, always build a complete result dict
        assessment = ""
        if algorithm == "Face Verification":
            if similarity > 70:
                assessment = "Same Person"
            elif similarity > 40:
                assessment = "Uncertain"
            elif similarity == 0:
                assessment = "No face detected"
            else:
                assessment = "Different Person"
        elif algorithm == "Machine Part Comparison":
            assessment = visual_label if visual_label else "Machine Part Comparison"
        elif algorithm == "Signature Verification":
            if similarity == 0:
                assessment = "No signature detected"
            else:
                assessment = visual_label
        else:
            assessment = visual_label
        # Ensure all keys are present
        results["comparisons"].append({
            "filename": comp_file.filename,
            "similarity_score": round(similarity, 2),
            "num_differences": int(num_diffs),
            "processed_image_url": processed_url if 'processed_url' in locals() else '',
            "visual_output": visual_url if 'visual_url' in locals() else '',
            "visual_label": visual_label,
            "assessment": assessment
        })
    results["overall_similarity"] = round(np.mean(similarities), 2) if similarities else 0
    return results, processed_paths

# At the end, always return URLs as /static/results/{sanitized_filename}
def url_with_leading_slash(path):
    if not path:
        return ""
    # Handle static/results paths
    if path.startswith("static/results/"):
        return "/static/results/" + os.path.basename(path)
    elif path.startswith("static/"):
        return "/" + path
    elif path.startswith("/static/"):
        return path
    else:
        return "/static/results/" + os.path.basename(path) 