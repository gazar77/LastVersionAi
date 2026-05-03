import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_vessel_diameter_map(vessel_mask: np.ndarray) -> np.ndarray:
    dt = distance_transform_edt(vessel_mask > 0)
    return dt * 2.0


def catheter_mm_per_pixel(
    catheter_width_px: float | None,
    catheter_diameter_mm: float = 2.2
) -> float | None:
    if catheter_width_px is None or catheter_width_px <= 0:
        return None
    return float(catheter_diameter_mm / catheter_width_px)


def px_to_mm(value_px: float, mm_per_pixel: float | None) -> float | None:
    if mm_per_pixel is None:
        return None
    return float(value_px * mm_per_pixel)


def classify_stenosis_severity(stenosis_percentage: float) -> str:
    if stenosis_percentage >= 99.5:
        return "Total Occlusion"
    if stenosis_percentage < 50:
        return "Mild"
    elif stenosis_percentage <= 70:
        return "Moderate"
    else:
        return "Severe"


def measure_stenosis(
    vessel_mask: np.ndarray,
    stenosis_mask: np.ndarray,
    centerline: np.ndarray,
    catheter_width_px: float | None = None,
    catheter_diameter_mm: float = 2.2
) -> list[dict]:
   
    results = []

    if vessel_mask is None or stenosis_mask is None or centerline is None:
        return results

    vessel_mask = (vessel_mask > 0).astype(np.uint8)
    stenosis_mask = (stenosis_mask > 0).astype(np.uint8)
    centerline = (centerline > 0).astype(np.uint8)

    diameter_map = compute_vessel_diameter_map(vessel_mask)

    lesion_labels, lesion_map, lesion_stats, _ = cv2.connectedComponentsWithStats(
        stenosis_mask, connectivity=8
    )

    centerline_points = np.column_stack(np.where(centerline > 0))
    if len(centerline_points) == 0:
        return results

    mm_per_pixel = catheter_mm_per_pixel(
        catheter_width_px=catheter_width_px,
        catheter_diameter_mm=catheter_diameter_mm
    )

    for i in range(1, lesion_labels):
        lesion_component = (lesion_map == i).astype(np.uint8)
        area = lesion_stats[i, cv2.CC_STAT_AREA]

        if area < 10:
            continue

        ys, xs = np.where(lesion_component > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        # centerline points inside lesion
        lesion_centerline = []
        for py, px in centerline_points:
            if lesion_component[py, px] > 0:
                lesion_centerline.append((py, px))

        # fallback: centerline points داخل البوكس
        if len(lesion_centerline) == 0:
            for py, px in centerline_points:
                if x1 <= px <= x2 and y1 <= py <= y2:
                    lesion_centerline.append((py, px))

        if len(lesion_centerline) == 0:
            continue

        local_diams = [diameter_map[py, px] for py, px in lesion_centerline if diameter_map[py, px] > 0]
        if len(local_diams) == 0:
            continue

        narrow_diameter_px = float(np.min(local_diams))
        lesion_length_px = float(len(lesion_centerline))

        if lesion_length_px < 8:
            continue

        # local reference diameter حول نفس الـ lesion
        reference_candidates = []
        for py, px in lesion_centerline:
            for dy in range(-20, 21):
                for dx in range(-20, 21):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < vessel_mask.shape[0] and 0 <= nx < vessel_mask.shape[1]:
                        if centerline[ny, nx] > 0:
                            d = diameter_map[ny, nx]
                            if d > 0:
                                reference_candidates.append(d)

        if len(reference_candidates) < 5:
            continue

        reference_diameter_px = float(np.percentile(reference_candidates, 80))

        if reference_diameter_px <= 0:
            continue

        if narrow_diameter_px >= reference_diameter_px:
            continue

        stenosis_percentage = (1 - (narrow_diameter_px / reference_diameter_px)) * 100
        stenosis_percentage = max(0.0, min(100.0, stenosis_percentage))

# تجاهل أي ضيق أقل من 50%
        if stenosis_percentage < 50:
           continue

        severity = classify_stenosis_severity(stenosis_percentage)

        

        results.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],

            "lesion_length_px": lesion_length_px,
            "narrow_diameter_px": narrow_diameter_px,
            "reference_diameter_px": reference_diameter_px,

            "mm_per_pixel": mm_per_pixel,
            "catheter_width_px": catheter_width_px,
            "catheter_diameter_mm": catheter_diameter_mm,

            "lesion_length_mm": px_to_mm(lesion_length_px, mm_per_pixel),
            "narrow_diameter_mm": px_to_mm(narrow_diameter_px, mm_per_pixel),
            "reference_diameter_mm": px_to_mm(reference_diameter_px, mm_per_pixel),

            "stenosis_percentage": float(stenosis_percentage),
            "severity": severity
        })

    return results