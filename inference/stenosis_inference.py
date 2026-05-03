import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_roi_for_stenosis(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    img = cv2.resize(image_rgb, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0).to(device)
    return img


def vessel_mask_to_roi_bbox(vessel_mask: np.ndarray, margin: int = 20):
    ys, xs = np.where(vessel_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = max(0, xs.min() - margin)
    y1 = max(0, ys.min() - margin)
    x2 = min(vessel_mask.shape[1], xs.max() + margin)
    y2 = min(vessel_mask.shape[0], ys.max() + margin)

    return [int(x1), int(y1), int(x2), int(y2)]


def mask_to_bboxes(mask: np.ndarray, min_area: int = 20):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])

    return bboxes


def run_stenosis_inference(image_rgb: np.ndarray,
                           vessel_mask: np.ndarray,
                           stenosis_model,
                           device: torch.device,
                           threshold: float = 0.5,
                           roi_margin: int = 20):
    roi_bbox = vessel_mask_to_roi_bbox(vessel_mask, margin=roi_margin)
    if roi_bbox is None:
        return {
            "roi_bbox": None,
            "stenosis_mask": np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            "bboxes": [],
            "roi_image": None,
        }

    x1, y1, x2, y2 = roi_bbox
    roi = image_rgb[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            "roi_bbox": roi_bbox,
            "stenosis_mask": np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            "bboxes": [],
            "roi_image": None,
        }

    inp = preprocess_roi_for_stenosis(roi, device)

    with torch.no_grad():
        logits = stenosis_model(inp)
        pred = torch.sigmoid(logits).cpu().numpy()[0, 0]

    pred_mask_roi = (pred > threshold).astype(np.uint8)

    pred_mask_resized = cv2.resize(
        pred_mask_roi,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST
    )

    full_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = pred_mask_resized

    bboxes = mask_to_bboxes(full_mask, min_area=10)

    return {
        "roi_bbox": roi_bbox,
        "stenosis_mask": full_mask,
        "bboxes": bboxes,
        "roi_image": roi,
    }