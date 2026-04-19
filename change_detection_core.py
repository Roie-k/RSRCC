import dataclasses
from typing import Iterator, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IOU_THRESH: float = 0.18
MIN_REGION_PIXELS: int = 500
MIN_CHANGED_PIXELS: int = 550
MIN_TOTAL_CHANGED_PIXELS: int = 490

MASK2FORMER_CLASSES: List[str] = [
    "building",
    "driveway",
    "parking_lot",
    "road_paved",
    "road_unpaved",
    "sidewalk",
    "trail",
    "tree",
    "water",
    "track",
    "bike_lane",
    "crosswalk",
    "painted_median",
    "railway_tracks",
    "shipping_container",
    "stairs",
    "swimming_pool",
    "athletic_field",
]

SEGFORMER_CLASSES: List[str] = [
    "background",
    "residential_area",
    "road",
    "river",
    "forest",
    "unused_land",
    "reserved",
]


@dataclasses.dataclass
class ChangeInstance:
    class_name: str
    class_id: int
    footprint: np.ndarray
    iou: float
    changed_px: int
    changed_ratio: float
    xmin: int
    ymin: int
    width: int
    height: int
    dominant_image_idx: int


class SatelliteSegmenter:
    """
    Segmentation wrapper supporting public Hugging Face backends.

    Supported backends:
    - mask2former: mfaytin/mask2former-satellite
    - segformer:   Pranilllllll/segformer-satellite-segementation
    """

    def __init__(
        self,
        backend: Literal["mask2former", "segformer"] = "mask2former",
        model_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if backend == "mask2former":
            self.model_id = model_id or "mfaytin/mask2former-satellite"
            self.class_names = MASK2FORMER_CLASSES
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.model_id
            ).to(self.device)
        elif backend == "segformer":
            self.model_id = model_id or "Pranilllllll/segformer-satellite-segementation"
            self.class_names = SEGFORMER_CLASSES
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_id
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.model.eval()

    @torch.no_grad()
    def segment(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if self.backend == "mask2former":
            outputs = self.model(**inputs)
            prediction = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],
            )[0]
            return prediction.cpu().numpy().astype(np.int64)

        if self.backend == "segformer":
            outputs = self.model(**inputs)
            logits = outputs.logits  # [B, C, h, w]
            upsampled_logits = F.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            prediction = upsampled_logits.argmax(dim=1)[0]
            return prediction.cpu().numpy().astype(np.int64)

        raise RuntimeError("Invalid backend state.")


class ChangeDetectionCore:
    """
    Candidate generation stage:
    1. segment both timestamps
    2. compute semantic difference mask
    3. extract connected components per class
    4. filter by region size, IoU, and changed-pixel statistics
    """

    def __init__(
        self,
        backend: Literal["mask2former", "segformer"] = "mask2former",
        model_id: str | None = None,
    ) -> None:
        self.segmenter = SatelliteSegmenter(backend=backend, model_id=model_id)
        self.class_names = self.segmenter.class_names
        self.backend = backend

    def run_inference(self, img1: Image.Image, img2: Image.Image) -> List[ChangeInstance]:
        mask1 = self.segmenter.segment(img1)
        mask2 = self.segmenter.segment(img2)

        diff_mask = mask1 != mask2
        if int(diff_mask.sum()) < MIN_TOTAL_CHANGED_PIXELS:
            return []

        return list(self._extract_instances(mask1, mask2, diff_mask))

    def _extract_instances(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        diff: np.ndarray,
    ) -> Iterator[ChangeInstance]:
        for class_id, class_name in enumerate(self.class_names):
            if class_name in {"background", "reserved"}:
                continue

            combined = (m1 == class_id) | (m2 == class_id)
            labeled, num_features = ndimage.label(combined)

            for component_idx in range(1, num_features + 1):
                region = labeled == component_idx
                region_pixels = int(region.sum())

                if region_pixels < MIN_REGION_PIXELS:
                    continue

                m1_region = (m1 == class_id) & region
                m2_region = (m2 == class_id) & region

                union_px = int((m1_region | m2_region).sum())
                inter_px = int((m1_region & m2_region).sum())
                iou = inter_px / (union_px + 1e-8)

                changed_px = int((region & diff).sum())
                changed_ratio = changed_px / (region_pixels + 1e-8)

                if iou >= IOU_THRESH:
                    continue
                if changed_px <= MIN_CHANGED_PIXELS:
                    continue

                coords = np.argwhere(region)
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)

                width = int(xmax - xmin + 1)
                height = int(ymax - ymin + 1)

                dominant_image_idx = 1 if int(m1_region.sum()) > int(m2_region.sum()) else 2

                yield ChangeInstance(
                    class_name=class_name,
                    class_id=class_id,
                    footprint=region,
                    iou=float(iou),
                    changed_px=changed_px,
                    changed_ratio=float(changed_ratio),
                    xmin=int(xmin),
                    ymin=int(ymin),
                    width=width,
                    height=height,
                    dominant_image_idx=dominant_image_idx,
                )
