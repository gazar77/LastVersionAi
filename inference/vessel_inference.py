import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image_for_vessel(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    img = cv2.resize(image_rgb, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0).to(device)
    return img


def refine_vessel_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    min_area = 60

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    cleaned = (cleaned > 0).astype(np.uint8)
    return cleaned


def thin_vessel_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    return skeleton.astype(np.uint8)


def run_vessel_inference(image_rgb: np.ndarray, vessel_model, device: torch.device, threshold: float = 0.5):
    inp = preprocess_image_for_vessel(image_rgb, device)

    with torch.no_grad():
        logits = vessel_model(inp)
        pred = torch.sigmoid(logits).cpu().numpy()[0, 0]

    pred_mask = (pred > threshold).astype(np.uint8)
    refined_mask = refine_vessel_mask(pred_mask)
    refined_mask = thin_vessel_mask(refined_mask)
    centerline = extract_centerline(refined_mask)

    return {
        "pred_mask": pred_mask,
        "refined_mask": refined_mask,
        "centerline": centerline,
    }