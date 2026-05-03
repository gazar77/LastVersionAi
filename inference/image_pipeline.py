import os
import json
import cv2
import numpy as np

from inference.model_loader import ModelLoader
from inference.vessel_inference import run_vessel_inference
from inference.stenosis_inference import run_stenosis_inference
from inference.measurements import measure_stenosis


class ImagePipeline:
    def __init__(self, project_root: str, device: str | None = None):
        self.project_root = project_root
        self.loader = ModelLoader(project_root=project_root, device=device)
        self.vessel_model, self.stenosis_model = self.loader.load_all()

        self.outputs_root = os.path.join(self.project_root, "outputs")
        self.overlay_dir = os.path.join(self.outputs_root, "overlays")
        self.mask_dir = os.path.join(self.outputs_root, "masks")
        self.report_dir = os.path.join(self.outputs_root, "reports")

        os.makedirs(self.overlay_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def analyze_image(
        self,
        image_path: str,
        output_prefix: str = "case",
        vessel_threshold: float = 0.5,
        stenosis_threshold: float = 0.4,
        roi_margin: int = 20,
        save_outputs: bool = True,
        catheter_width_px: float | None = None,
        catheter_diameter_mm: float = 2.2,
    ) -> dict:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        vessel_result = run_vessel_inference(
            image_rgb=image_rgb,
            vessel_model=self.vessel_model,
            device=self.loader.device,
            threshold=vessel_threshold,
        )

        stenosis_result = run_stenosis_inference(
            image_rgb=image_rgb,
            vessel_mask=vessel_result["refined_mask"],
            stenosis_model=self.stenosis_model,
            device=self.loader.device,
            threshold=stenosis_threshold,
            roi_margin=roi_margin,
        )

        measurement_results = measure_stenosis(
            vessel_mask=vessel_result["refined_mask"],
            stenosis_mask=stenosis_result["stenosis_mask"],
            centerline=vessel_result["centerline"],
            catheter_width_px=catheter_width_px,
            catheter_diameter_mm=catheter_diameter_mm,
        )

        overlay = image_rgb.copy()

        # vessel in green
        green = np.zeros_like(overlay)
        green[:, :, 1] = vessel_result["refined_mask"] * 255
        overlay = cv2.addWeighted(overlay, 1.0, green, 0.25, 0)

        # stenosis in red
        red = np.zeros_like(overlay)
        red[:, :, 0] = stenosis_result["stenosis_mask"] * 255
        overlay = cv2.addWeighted(overlay, 1.0, red, 0.35, 0)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # draw boxes + only stenosis % on image
        for item in measurement_results:
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

            stenosis_text = f"{item['stenosis_percentage']:.1f}%"
            text_y = max(y1 - 10, 20)

            cv2.putText(
                overlay_bgr,
                stenosis_text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        final_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        saved_paths = {}
        if save_outputs:
            overlay_path = os.path.join(self.overlay_dir, f"{output_prefix}_overlay.png")
            vessel_mask_path = os.path.join(self.mask_dir, f"{output_prefix}_vessel_mask.png")
            stenosis_mask_path = os.path.join(self.mask_dir, f"{output_prefix}_stenosis_mask.png")
            report_path = os.path.join(self.report_dir, f"{output_prefix}_report.json")

            cv2.imwrite(overlay_path, cv2.cvtColor(final_overlay_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(vessel_mask_path, vessel_result["refined_mask"] * 255)
            cv2.imwrite(stenosis_mask_path, stenosis_result["stenosis_mask"] * 255)

            with open(report_path, "w") as f:
                json.dump(measurement_results, f, indent=2)

            saved_paths = {
                "overlay_path": overlay_path,
                "vessel_mask_path": vessel_mask_path,
                "stenosis_mask_path": stenosis_mask_path,
                "report_path": report_path,
            }

        return {
            "image_path": image_path,
            "roi_bbox": stenosis_result["roi_bbox"],
            "stenosis_boxes": stenosis_result["bboxes"],
            "measurement_results": measurement_results,
            "saved_paths": saved_paths,
        }