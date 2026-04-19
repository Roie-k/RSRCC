import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BestOfNVerifier:
    """
    Retrieval-augmented Gemma 3 verifier for semantic class validation on
    localized satellite image patches.

    This module follows the paper's reward-style verification setup:
    - retrieve few-shot examples for the queried class
    - present examples + query patch to Gemma 3
    - extract a scalar score in {1,2,3,4,5}

    Note:
        This implements the paper's prompt-based patch scoring formulation.
        It does not implement a full multi-hypothesis Best-of-N search loop by itself.
        Higher-level orchestration can call this scorer across candidate hypotheses.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        max_examples: int = 5,
        patch_size: Tuple[int, int] = (224, 224),
        do_pan_and_scan: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_examples = max_examples
        self.patch_size = patch_size
        self.do_pan_and_scan = do_pan_and_scan

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info("Loading Gemma 3 model: %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=self.dtype,
        ).eval()
        logger.info("Gemma 3 loaded successfully.")

    @staticmethod
    def _safe_int(value: Any) -> int:
        return int(round(float(value)))

    def _get_extended_patch(
        self,
        image: Image.Image,
        xmin: int,
        ymin: int,
        width: int,
        height: int,
        expansion_ratio: float = 0.5,
    ) -> Image.Image:
        """
        Crop a patch with context expansion around the original box.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_width, img_height = image.size

        x_expand = int(np.ceil(width * expansion_ratio))
        y_expand = int(np.ceil(height * expansion_ratio))

        x0 = max(0, xmin - x_expand)
        y0 = max(0, ymin - y_expand)
        x1 = min(img_width, xmin + width + x_expand)
        y1 = min(img_height, ymin + height + y_expand)

        return image.crop((x0, y0, x1, y1))

    def _load_image_from_row(self, row: pd.Series) -> Image.Image:
        """
        Load a PIL image from a dataframe row.

        Expected:
        - row['image_bytes'] contains raw bytes
        """
        raw = row["image_bytes"]
        if isinstance(raw, bytes):
            image = Image.open(BytesIO(raw)).convert("RGB")
            return image

        raise TypeError(
            "Unsupported image storage format in examples_df['image_bytes']. "
            "Expected raw bytes."
        )

    def _get_images_by_class(
        self,
        query_class: str,
        examples_df: pd.DataFrame,
    ) -> Tuple[List[Image.Image], List[int]]:
        """
        Retrieve few-shot example patches and their numeric scores for a semantic class.
        """
        required_cols = {"Class", "Score", "image_bytes", "xmin", "ymin", "width", "height"}
        missing = required_cols - set(examples_df.columns)
        if missing:
            raise ValueError(f"examples_df is missing required columns: {sorted(missing)}")

        class_df = examples_df[examples_df["Class"] == query_class].head(self.max_examples)

        images: List[Image.Image] = []
        scores: List[int] = []

        for _, row in class_df.iterrows():
            full_image = self._load_image_from_row(row)
            patch = self._get_extended_patch(
                image=full_image,
                xmin=self._safe_int(row["xmin"]),
                ymin=self._safe_int(row["ymin"]),
                width=self._safe_int(row["width"]),
                height=self._safe_int(row["height"]),
            ).resize(self.patch_size)

            images.append(patch)
            scores.append(self._safe_int(row["Score"]))

        return images, scores

    @staticmethod
    def _build_paper_prompt(selected_class: str) -> str:
        """
        Exact paper-aligned scoring instruction.
        """
        return (
            "You are an expert in recognizing objects from satellite images. "
            f"Your task is to score a query image patch from 1 to 5. "
            f"You need to specify if {selected_class} appears in the image. "
            "All images are satellite images. Return only the numerical score "
            "(1, 2, 3, 4, or 5).\n\n"
            f"5: There is definitely a {selected_class} in the last image. "
            "The object's shape, shadow, and features are clearly visible from above.\n"
            f"4: Very likely that the image contains a {selected_class}. "
            "Features are mostly clear.\n"
            f"3: Probably the image contains a {selected_class}, but visibility or details are ambiguous.\n"
            f"2: Unlikely that a {selected_class} appears in the image.\n"
            f"1: Definitely does not contain a {selected_class}."
        )

    def _build_messages_and_images(
        self,
        selected_class: str,
        support_images: List[Image.Image],
        support_scores: List[int],
        query_patch: Image.Image,
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """
        Build Gemma 3 multimodal messages and aligned image list.

        Important:
            For Gemma 3, every {'type': 'image'} slot in the chat content must
            correspond to an image in the images list passed to apply_chat_template().
        """
        prompt_text = self._build_paper_prompt(selected_class)

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        image_batch: List[Image.Image] = []

        for idx, (img, score) in enumerate(zip(support_images, support_scores), start=1):
            content.extend(
                [
                    {"type": "text", "text": f"Example ({idx}):"},
                    {"type": "image"},
                    {"type": "text", "text": f"Score = {score}"},
                ]
            )
            image_batch.append(img)

        content.extend(
            [
                {"type": "text", "text": f"Example ({len(support_images) + 1}):"},
                {"type": "image"},
                {"type": "text", "text": "Score = ?"},
            ]
        )
        image_batch.append(query_patch)

        messages = [{"role": "user", "content": content}]
        return messages, image_batch

    @staticmethod
    def _extract_score(response: str) -> Optional[int]:
        """
        Extract the first integer score in [1, 5] from the model response.
        """
        match = re.search(r"\b([1-5])\b", response)
        if match:
            return int(match.group(1))
        return None

    def run_verification(
        self,
        img1: Image.Image,
        img2: Image.Image,
        change_metadata: Dict[str, Any],
        image_num_to_verify: int,
        examples_df: pd.DataFrame,
        max_new_tokens: int = 16,
    ) -> Optional[int]:
        """
        Score whether the target semantic class appears in a localized patch.

        Args:
            img1: First temporal image.
            img2: Second temporal image.
            change_metadata: Must contain:
                - class_name
                - xmin
                - ymin
                - width
                - height
            image_num_to_verify:
                - 1 => verify patch from img1
                - 2 => verify patch from img2
            examples_df: Few-shot examples dataframe.
            max_new_tokens: Generation length for Gemma output.

        Returns:
            Integer score in [1, 5], or None if parsing fails.
        """
        required_keys = {"class_name", "xmin", "ymin", "width", "height"}
        missing = required_keys - set(change_metadata.keys())
        if missing:
            raise ValueError(f"change_metadata missing required keys: {sorted(missing)}")

        selected_class = str(change_metadata["class_name"])
        logger.info("Running Gemma verification for class '%s'", selected_class)

        support_images, support_scores = self._get_images_by_class(
            query_class=selected_class,
            examples_df=examples_df,
        )

        if len(support_images) == 0:
            logger.warning("No few-shot examples found for class '%s'", selected_class)

        base_image = img1 if image_num_to_verify == 1 else img2
        query_patch = self._get_extended_patch(
            image=base_image.convert("RGB"),
            xmin=self._safe_int(change_metadata["xmin"]),
            ymin=self._safe_int(change_metadata["ymin"]),
            width=self._safe_int(change_metadata["width"]),
            height=self._safe_int(change_metadata["height"]),
        ).resize(self.patch_size)

        messages, image_batch = self._build_messages_and_images(
            selected_class=selected_class,
            support_images=support_images,
            support_scores=support_scores,
            query_patch=query_patch,
        )

        # Gemma 3 multimodal processing:
        # pass both the structured messages and the actual images list.
        inputs = self.processor.apply_chat_template(
            messages,
            images=image_batch,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            do_pan_and_scan=self.do_pan_and_scan,
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        score = self._extract_score(response)
        if score is None:
            logger.warning("Could not parse score from Gemma response: %s", response)
            return None

        logger.info("Gemma score for class '%s': %d", selected_class, score)
        return score
