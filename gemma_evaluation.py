import argparse
import io
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from transformers import pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SUPPORTED_GEMMA_MODELS = {
    "4b": "google/gemma-3-4b-it",
    "27b": "google/gemma-3-27b-it",
}


class GemmaEvaluator:
    """
    Evaluate final benchmark samples using Gemma 3 multimodal models.

    Supported task types:
    - yes_no
    - mcq
    """

    def __init__(self, model_id: str = "google/gemma-3-4b-it") -> None:
        logger.info("Initializing Gemma evaluator with model %s", model_id)

        self.model_id = model_id
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=False,
        )

    @staticmethod
    def _hex_to_image(image_hex: str) -> Image.Image:
        image_bytes = bytes.fromhex(image_hex)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def _extract_generated_text(output: Any) -> str:
        generated = output[0]["generated_text"]

        if isinstance(generated, list) and len(generated) > 0:
            last_turn = generated[-1]
            if isinstance(last_turn, dict):
                content = last_turn.get("content", "")
                if isinstance(content, str):
                    return content.strip()

        return str(generated).strip()

    @staticmethod
    def _parse_yes_no(text: str) -> Optional[str]:
        match = re.search(r"\b(Yes|No)\b", text, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).capitalize()

    @staticmethod
    def _parse_mcq(text: str) -> Optional[str]:
        match = re.search(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).upper()

    def _build_messages(self, row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        task_type = row.get("task_type")
        question = row.get("question")

        if not question:
            return None

        if task_type == "yes_no":
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Answer the user's question about the two images. "
                                "Return only Yes or No."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image 1 (Before):"},
                        {"type": "image"},
                        {"type": "text", "text": "Image 2 (After):"},
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]

        if task_type == "mcq":
            options = row.get("options")
            if not isinstance(options, dict):
                return None
            if not all(k in options for k in ("A", "B", "C", "D")):
                return None

            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Answer the user's multiple-choice question about the two images. "
                                "Return only the correct letter: A, B, C, or D."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image 1 (Before):"},
                        {"type": "image"},
                        {"type": "text", "text": "Image 2 (After):"},
                        {"type": "image"},
                        {"type": "text", "text": question},
                        {"type": "text", "text": f"A) {options['A']}"},
                        {"type": "text", "text": f"B) {options['B']}"},
                        {"type": "text", "text": f"C) {options['C']}"},
                        {"type": "text", "text": f"D) {options['D']}"},
                    ],
                },
            ]

        return None

    def evaluate_sample(self, row: Dict[str, Any]) -> Optional[str]:
        task_type = row.get("task_type")
        before_image_hex = row.get("before_image_hex")
        after_image_hex = row.get("after_image_hex")

        if not before_image_hex or not after_image_hex:
            return None

        messages = self._build_messages(row)
        if messages is None:
            return None

        before_img = self._hex_to_image(before_image_hex)
        after_img = self._hex_to_image(after_image_hex)

        output = self.pipe(
            text=messages,
            images=[before_img, after_img],
            max_new_tokens=8,
            generate_kwargs={"do_sample": False},
        )

        raw_output = self._extract_generated_text(output)

        if task_type == "yes_no":
            return self._parse_yes_no(raw_output)
        if task_type == "mcq":
            return self._parse_mcq(raw_output)

        return None

    def run_benchmark(self, input_path: str, output_path: str) -> None:
        logger.info("Running Gemma benchmark on %s", input_path)

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
            rows = df.to_dict(orient="records")
        else:
            df = pd.read_json(input_path, lines=True)
            rows = df.to_dict(orient="records")

        results: List[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            sample_id = row.get("sample_id", idx)
            task_type = row.get("task_type", "unknown")
            logger.info("Evaluating sample %s (task=%s)", sample_id, task_type)

            prediction = self.evaluate_sample(row)

            results.append(
                {
                    "sample_id": sample_id,
                    "task_type": task_type,
                    "ground_truth": row.get("answer"),
                    "prediction": prediction,
                    "is_correct": prediction == row.get("answer") if prediction is not None else False,
                }
            )

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logger.info("Benchmark finished. Results saved to %s", output_path)


def resolve_model_name(model_arg: str) -> str:
    lowered = model_arg.strip().lower()
    return SUPPORTED_GEMMA_MODELS.get(lowered, model_arg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-based benchmark evaluation")
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        help="Model identifier or shorthand: 4b / 27b",
    )
    parser.add_argument("--input", required=True, help="Input CSV/JSONL path")
    parser.add_argument("--output", required=True, help="Output results CSV path")
    args = parser.parse_args()

    model_id = resolve_model_name(args.model)
    evaluator = GemmaEvaluator(model_id=model_id)
    evaluator.run_benchmark(args.input, args.output)


if __name__ == "__main__":
    main()
