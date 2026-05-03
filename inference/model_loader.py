import os
from typing import Tuple

import torch
import segmentation_models_pytorch as smp


class ModelLoader:
    def __init__(self, project_root: str, device: str | None = None):
        self.project_root = project_root
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.vessel_model_path = os.path.join(
            self.project_root, "models", "vessel", "best_vessel_model_run2.pth"
        )
        self.stenosis_model_path = os.path.join(
            self.project_root, "models", "stenosis", "best_stenosis_model_final.pth"
        )

        self.vessel_model = None
        self.stenosis_model = None

    def _build_vessel_model(self) -> torch.nn.Module:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)
        return model

    def _build_stenosis_model(self) -> torch.nn.Module:
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)
        return model

    def load_vessel_model(self) -> torch.nn.Module:
        if self.vessel_model is not None:
            return self.vessel_model

        if not os.path.exists(self.vessel_model_path):
            raise FileNotFoundError(f"Vessel model not found: {self.vessel_model_path}")

        model = self._build_vessel_model()
        state_dict = torch.load(self.vessel_model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.vessel_model = model
        return self.vessel_model

    def load_stenosis_model(self) -> torch.nn.Module:
        if self.stenosis_model is not None:
            return self.stenosis_model

        if not os.path.exists(self.stenosis_model_path):
            raise FileNotFoundError(f"Stenosis model not found: {self.stenosis_model_path}")

        model = self._build_stenosis_model()
        state_dict = torch.load(self.stenosis_model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.stenosis_model = model
        return self.stenosis_model

    def load_all(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        vessel_model = self.load_vessel_model()
        stenosis_model = self.load_stenosis_model()
        return vessel_model, stenosis_model