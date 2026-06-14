"""
Grounding DINO wrapper for open-vocabulary object detection.

Provides:
  - GroundingDINOWrapper: load model, generate boxes from text prompts
  - boxes_original_to_sam: coordinate scaling utility

Usage:
    wrapper = GroundingDINOWrapper(gd_ckpt_path, gd_config_path)
    boxes = wrapper.generate_boxes(pil_image, "waterbody")
    # boxes shape: (N, 4), values: [x1, y1, x2, y2] in original image pixels
"""

try:
    import groundingdino
    from groundingdino.util.inference import load_model, predict
    from groundingdino.datasets import transforms as GD_transforms
    _GROUNDING_DINO_AVAILABLE = True
except ImportError:
    _GROUNDING_DINO_AVAILABLE = False

import logging
import numpy as np
import torch
from PIL import Image


class GroundingDINOWrapper:
    """Wrapper for Grounding DINO model loading and inference."""

    def __init__(
        self,
        gd_ckpt_path: str,
        gd_config_path: str,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        if not _GROUNDING_DINO_AVAILABLE:
            raise RuntimeError(
                "GroundingDINO is not installed.\n"
                "Install from source:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO.git\n"
                "  cd GroundingDINO && pip install -e ."
            )

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        logging.info(f"Loading Grounding DINO from {gd_ckpt_path}")
        self.model = load_model(gd_config_path, gd_ckpt_path)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def generate_boxes(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = None,
        text_threshold: float = None,
    ) -> torch.Tensor:
        """Run Grounding DINO on a PIL image and return bounding boxes.

        Args:
            image: PIL Image in RGB mode.
            text_prompt: Text prompt (e.g. "waterbody").
            box_threshold: Override default box confidence threshold.
            text_threshold: Override default text confidence threshold.

        Returns:
            torch.Tensor of shape (N, 4) with [x1, y1, x2, y2]
            in original image pixel coordinates.
            Returns a full-image box [0, 0, W, H] if no detections.
        """
        bt = box_threshold if box_threshold is not None else self.box_threshold
        tt = text_threshold if text_threshold is not None else self.text_threshold

        w, h = image.size

        # Grounding DINO prediction expects preprocessed tensor, not raw numpy
        transform = GD_transforms.Compose(
            [
                GD_transforms.RandomResize([800], max_size=1333),
                GD_transforms.ToTensor(),
                GD_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor, _ = transform(image.convert("RGB"), None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=img_tensor,
            caption=text_prompt,
            box_threshold=bt,
            text_threshold=tt,
        )

        if boxes.shape[0] == 0:
            logging.warning(
                f"GD found no objects for '{text_prompt}', using full-image box"
            )
            return torch.tensor([[0, 0, w, h]], dtype=torch.float32)

        # Convert from normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2).clamp(min=0) * w
        y1 = (cy - bh / 2).clamp(min=0) * h
        x2 = (cx + bw / 2).clamp(max=1) * w
        y2 = (cy + bh / 2).clamp(max=1) * h

        result = torch.stack([x1, y1, x2, y2], dim=1)
        return result

    @staticmethod
    def boxes_original_to_sam(
        boxes: torch.Tensor,
        orig_size: tuple,
        sam_size: int = 1024,
    ) -> torch.Tensor:
        """Scale boxes from original image coordinates to SAM's input frame.

        Args:
            boxes: (N, 4) tensor with [x1, y1, x2, y2] in original pixels.
            orig_size: (orig_w, orig_h) tuple.
            sam_size: SAM's input image size (default 1024).

        Returns:
            (N, 4) tensor with coordinates scaled to sam_size x sam_size.
        """
        orig_w, orig_h = orig_size
        scale_x = sam_size / orig_w
        scale_y = sam_size / orig_h
        scaled = boxes.clone()
        scaled[:, 0] *= scale_x  # x1
        scaled[:, 1] *= scale_y  # y1
        scaled[:, 2] *= scale_x  # x2
        scaled[:, 3] *= scale_y  # y2
        return scaled
