import argparse
import io
import logging
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
from google import genai
from google.genai import types


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GeminiEvaluator:
    """
    Evaluate final benchmark samples using Gemini 2.5 Flash.

    Supported task types:
    - yes_no
    - mcq
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.5-flash",
        sleep_seconds: float = 0.5,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.sleep_seconds = sleep_seconds

    @staticmethod
    def _hex_to_image(image_hex: str) -> Image.Image:
        image_bytes = bytes.fromhex(image_hex)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

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

    def _build_prompt(self, row: Dict[str, Any]) -> Optional[str]:
        task_type = row.get("task_type")
        question = row.get("question")

        if not question:
            return None

        if task_type == "yes_no":
            return (
                "Answer the user's question about the two images. "
                "Return only Yes or No.\n\n"
                f"{question}"
            )

        if task_type == "mcq":
            options = row.get("options")
            if not isinstance(options, dict):
                return None
            if not all(k in options for k in ("A", "B", "C", "D")):
                return None

            return (
                "Answer the user's multiple-choice question about the two images. "
                "Return only the correct letter: A, B, C, or D.\n\n"
                f"{question}\n"
                f"A) {options['A']}\n"
                f"B) {options['B']}\n"
                f"C) {options['C']}\n"
                f"D) {options['D']}"
            )

        return None

    def _call_gemini(
        self,
        before_image_hex: str,
        after_image_hex: str,
        prompt: str,
    ) -> Optional[str]:
        before_img = self._hex_to_image(before_image_hex)
        after_img = self._hex_to_image(after_image_hex)

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    "Image 1 (Before):",
                    before_img,
                    "Image 2 (After):",
                    after_img,
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8,
                ),
            )
            if response.text is None:
                return None
            return response.text.strip()
        except Exception as exc:
            logger.error("Gemini evaluation error: %s", exc)
            return None

    def evaluate_sample(self, row: Dict[str, Any]) -> Optional[str]:
        task_type = row.get("task_type")
        before_image_hex = row.get("before_image_hex")
        after_image_hex = row.get("after_image_hex")

        if not before_image_hex or not after_image_hex:
            return None

        prompt = self._build_prompt(row)
        if prompt is None:
            return None

        raw_output = self._call_gemini(
            before_image_hex=before_image_hex,
            after_image_hex=after_image_hex,
            prompt=prompt,
        )
        if raw_output is None:
            return None

        if task_type == "yes_no":
            return self._parse_yes_no(raw_output)
        if task_type == "mcq":
            return self._parse_mcq(raw_output)

        return None

    def run_benchmark(self, input_path: str, output_path: str) -> None:
        logger.info("Running Gemini benchmark on %s", input_path)

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

            time.sleep(self.sleep_seconds)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logger.info("Benchmark complete. Results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini-based benchmark evaluation")
    parser.add_argument("--api_key", required=True, help="Google GenAI API key")
    parser.add_argument("--input_data", required=True, help="Input CSV/JSONL path")
    parser.add_argument("--output_results", required=True, help="Output results CSV path")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model version")
    parser.add_argument("--sleep_seconds", type=float, default=0.5, help="Delay between API calls")
    args = parser.parse_args()

    evaluator = GeminiEvaluator(
        api_key=args.api_key,
        model_id=args.model,
        sleep_seconds=args.sleep_seconds,
    )
    evaluator.run_benchmark(args.input_data, args.output_results)


if __name__ == "__main__":
    main()
