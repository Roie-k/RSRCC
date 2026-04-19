import argparse
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from PIL import Image
from google import genai
from google.genai import types


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GeminiDatasetGenerator:
    """
    Generate final benchmark samples from prompt-construction requests using Gemini 2.5 Flash.

    Input:
        JSONL file where each line contains:
        - sample_id
        - task_type: "yes_no" or "mcq"
        - before_image_hex
        - after_image_hex
        - prompt
        - metadata

    Output:
        JSONL file where each line contains a structured benchmark sample with:
        - sample_id
        - task_type
        - before_image_hex
        - after_image_hex
        - question
        - answer
        - optional options for MCQ
        - metadata

    Notes:
        - Open-ended generation is intentionally not supported.
        - By default, samples with answer "I am not sure" are discarded.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.5-flash",
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_output_tokens: int = 512,
        keep_uncertain: bool = False,
        sleep_seconds: float = 1.0,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.keep_uncertain = keep_uncertain
        self.sleep_seconds = sleep_seconds

    @staticmethod
    def _hex_to_image(image_hex: str) -> Image.Image:
        image_bytes = bytes.fromhex(image_hex)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            text = response.text
            if text is None:
                return None
            return text.strip()
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            return None

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"[ \t]+", " ", text.strip())

    def _parse_yes_no_output(self, text: str) -> Optional[Dict[str, str]]:
        question_match = re.search(
            r"\*\*Question:\*\*\s*(.+?)(?=\n\*\*Answer:\*\*|\Z)",
            text,
            flags=re.DOTALL,
        )
        answer_match = re.search(
            r"\*\*Answer:\*\*\s*(.+?)\s*$",
            text,
            flags=re.DOTALL,
        )

        if question_match is None or answer_match is None:
            question_match = re.search(
                r"Question:\s*(.+?)(?=\nAnswer:|\Z)",
                text,
                flags=re.DOTALL,
            )
            answer_match = re.search(
                r"Answer:\s*(.+?)\s*$",
                text,
                flags=re.DOTALL,
            )

        if question_match is None or answer_match is None:
            return None

        question = self._normalize_whitespace(question_match.group(1))
        answer = self._normalize_whitespace(answer_match.group(1))

        if answer.lower() not in {"yes", "no", "i am not sure"}:
            return None

        return {
            "question": question,
            "answer": answer,
        }

    def _parse_mcq_output(self, text: str) -> Optional[Dict[str, Any]]:
        question_match = re.search(
            r"\*\*Question:\*\*\s*(.+?)(?=\n\*\*A\)\*\*|\nA\)|\Z)",
            text,
            flags=re.DOTALL,
        )
        if question_match is None:
            question_match = re.search(
                r"Question:\s*(.+?)(?=\nA\)|\Z)",
                text,
                flags=re.DOTALL,
            )

        option_patterns = {
            "A": [r"\*\*A\)\*\*\s*(.+?)(?=\n\*\*B\)\*\*|\nB\)|\Z)", r"A\)\s*(.+?)(?=\nB\)|\Z)"],
            "B": [r"\*\*B\)\*\*\s*(.+?)(?=\n\*\*C\)\*\*|\nC\)|\Z)", r"B\)\s*(.+?)(?=\nC\)|\Z)"],
            "C": [r"\*\*C\)\*\*\s*(.+?)(?=\n\*\*D\)\*\*|\nD\)|\Z)", r"C\)\s*(.+?)(?=\nD\)|\Z)"],
            "D": [r"\*\*D\)\*\*\s*(.+?)(?=\n\*\*Answer:\*\*|\nAnswer:|\Z)", r"D\)\s*(.+?)(?=\nAnswer:|\Z)"],
        }

        options: Dict[str, str] = {}
        for key, patterns in option_patterns.items():
            matched = None
            for pattern in patterns:
                matched = re.search(pattern, text, flags=re.DOTALL)
                if matched is not None:
                    break
            if matched is None:
                return None
            options[key] = self._normalize_whitespace(matched.group(1))

        answer_match = re.search(
            r"\*\*Answer:\*\*\s*(.+?)\s*$",
            text,
            flags=re.DOTALL,
        )
        if answer_match is None:
            answer_match = re.search(
                r"Answer:\s*(.+?)\s*$",
                text,
                flags=re.DOTALL,
            )

        if question_match is None or answer_match is None:
            return None

        question = self._normalize_whitespace(question_match.group(1))
        answer = self._normalize_whitespace(answer_match.group(1))

        valid_answers = {"A", "B", "C", "D", "I am not sure"}
        if answer not in valid_answers:
            return None

        return {
            "question": question,
            "options": options,
            "answer": answer,
        }

    def _parse_response(self, task_type: str, text: str) -> Optional[Dict[str, Any]]:
        if task_type == "yes_no":
            return self._parse_yes_no_output(text)
        if task_type == "mcq":
            return self._parse_mcq_output(text)
        return None

    def _should_keep_sample(self, parsed: Dict[str, Any]) -> bool:
        answer = parsed.get("answer", "")
        if answer == "I am not sure" and not self.keep_uncertain:
            return False
        return True

    def _build_output_record(
        self,
        request_row: Dict[str, Any],
        parsed: Dict[str, Any],
        raw_output: str,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "sample_id": request_row["sample_id"],
            "task_type": request_row["task_type"],
            "before_image_hex": request_row["before_image_hex"],
            "after_image_hex": request_row["after_image_hex"],
            "question": parsed["question"],
            "answer": parsed["answer"],
            "metadata": request_row.get("metadata", {}),
            "generator_model": self.model_id,
            "gemini_raw_output": raw_output,
        }

        if request_row["task_type"] == "mcq":
            record["options"] = parsed["options"]

        return record

    def process_requests_jsonl(
        self,
        input_jsonl: str,
        output_jsonl: str,
    ) -> None:
        logger.info("Reading generation requests from %s", input_jsonl)

        total = 0
        kept = 0
        failed = 0
        filtered_uncertain = 0

        with open(input_jsonl, "r", encoding="utf-8") as fin, open(
            output_jsonl, "w", encoding="utf-8"
        ) as fout:
            for line_idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                total += 1

                try:
                    request_row = json.loads(line)
                except Exception as exc:
                    failed += 1
                    logger.error("Failed to parse JSONL line %d: %s", line_idx, exc)
                    continue

                task_type = request_row.get("task_type")
                if task_type not in {"yes_no", "mcq"}:
                    failed += 1
                    logger.warning(
                        "Skipping unsupported task_type '%s' at line %d",
                        task_type,
                        line_idx,
                    )
                    continue

                prompt = request_row.get("prompt")
                before_image_hex = request_row.get("before_image_hex")
                after_image_hex = request_row.get("after_image_hex")

                if not prompt or not before_image_hex or not after_image_hex:
                    failed += 1
                    logger.warning("Missing required fields at line %d", line_idx)
                    continue

                logger.info(
                    "Generating sample %s (task=%s)",
                    request_row.get("sample_id", f"line_{line_idx}"),
                    task_type,
                )

                raw_output = self._call_gemini(
                    before_image_hex=before_image_hex,
                    after_image_hex=after_image_hex,
                    prompt=prompt,
                )

                if raw_output is None:
                    failed += 1
                    continue

                parsed = self._parse_response(task_type=task_type, text=raw_output)
                if parsed is None:
                    failed += 1
                    logger.warning("Failed to parse Gemini output at line %d", line_idx)
                    logger.warning("Raw output: %s", raw_output)
                    time.sleep(self.sleep_seconds)
                    continue

                if not self._should_keep_sample(parsed):
                    filtered_uncertain += 1
                    logger.info(
                        "Filtered uncertain sample %s",
                        request_row.get("sample_id", f"line_{line_idx}"),
                    )
                    time.sleep(self.sleep_seconds)
                    continue

                output_record = self._build_output_record(
                    request_row=request_row,
                    parsed=parsed,
                    raw_output=raw_output,
                )
                fout.write(json.dumps(output_record) + "\n")
                kept += 1

                time.sleep(self.sleep_seconds)

        logger.info("Generation complete.")
        logger.info("Total requests: %d", total)
        logger.info("Kept samples: %d", kept)
        logger.info("Filtered uncertain: %d", filtered_uncertain)
        logger.info("Failed samples: %d", failed)
        logger.info("Saved final dataset to %s", output_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini-based RSRCC benchmark generator")
    parser.add_argument("--api_key", required=True, help="Google GenAI API key")
    parser.add_argument("--input_jsonl", required=True, help="Path to prompt requests JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Path to final benchmark JSONL")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument(
        "--keep_uncertain",
        action="store_true",
        help='Keep samples whose final answer is "I am not sure"',
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=1.0,
        help="Delay between API calls",
    )
    args = parser.parse_args()

    generator = GeminiDatasetGenerator(
        api_key=args.api_key,
        model_id=args.model,
        keep_uncertain=args.keep_uncertain,
        sleep_seconds=args.sleep_seconds,
    )
    generator.process_requests_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
    )


if __name__ == "__main__":
    main()
