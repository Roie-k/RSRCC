"""
Library for generating VLM instruction-tuning data using the official Gemini API.

This script takes the Best-of-N validated changes and utilizes Gemini 3 Flash 
to generate spatially-aware, complex questions and answers, ensuring 
strict adherence to the 'No Leakage' and 'Strict Prohibition' guidelines.
"""

import json
import logging
import io
import time
import argparse
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Optional, Iterator
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiDatasetGenerator:
    """
    Interfaces with the Gemini API to transform satellite metadata into 
    conversational instruction-tuning samples.
    """

    def __init__(self, api_key: str, model_id: str = "gemini-3-flash-preview"):
        """
        Initializes the Gemini client.

        Args:
            api_key: Your official Google GenAI API key.
            model_id: The specific Gemini model version to use.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def _prepare_image(self, img_bytes: bytes) -> Image.Image:
        """Converts raw bytes to a PIL Image for the GenAI SDK."""
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def generate_sample(
        self, 
        img1_bytes: bytes, 
        img2_bytes: bytes, 
        prompt: str
    ) -> Optional[str]:
        """
        Calls the Gemini API with a multimodal payload (2 images + 1 text prompt).
        """
        img1: Image.Image = self._prepare_image(img1_bytes)
        img2: Image.Image = self._prepare_image(img2_bytes)

        try:
            # Official Gemini API multimodal call
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    "Image 1 (Before):", img1,
                    "Image 2 (After):", img2,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    top_p=0.9,
                    max_output_tokens=512,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    def process_metadata_csv(self, input_csv: str, output_jsonl: str):
        """
        Iterates through the Best-of-N validated CSV and generates the final dataset.
        
        Args:
            input_csv: Path to the CSV containing validated change metadata.
            output_jsonl: Path to save the final Gemini-enhanced training data.
        """
        df: pd.DataFrame = pd.read_csv(input_csv)
        logger.info(f"Starting Gemini generation for {len(df)} rows.")

        with open(output_jsonl, 'w') as f:
            for idx, row in df.iterrows():
                # Extract image data (assuming stored as hex or direct bytes in the refactored CSV)
                img1_bytes: bytes = bytes.fromhex(row['img1_hex'])
                img2_bytes: bytes = bytes.fromhex(row['img2_hex'])
                
                # Retrieve the specialized prompt constructed in the previous stage
                # (e.g., the 'No Leakage' spatial reasoning prompts)
                prompt: str = row['generated_instruction_prompt']

                logger.info(f"Processing sample {idx}...")
                
                # Execute API Call
                gemini_output: Optional[str] = self.generate_sample(img1_bytes, img2_bytes, prompt)
                
                if gemini_output:
                    # Construct the final dataset entry
                    sample = {
                        "row_index": idx,
                        "metadata": {
                            "class": row.get('change_class_name', 'no_change'),
                            "vlm_reward_score": row.get('condition_flag', 0)
                        },
                        "gemini_raw_output": gemini_output
                    }
                    f.write(json.dumps(sample) + '\n')
                    
                # Rate limiting precaution for API stability
                time.sleep(1.0) 

        logger.info(f"Dataset generation complete. Saved to {output_jsonl}")

# --- CLI Implementation ---

def main():
    parser = argparse.ArgumentParser(description="Gemini-based Dataset Generation Pipeline")
    parser.add_argument("--api_key", required=True, help="Official Google GenAI API Key")
    parser.add_argument("--input_csv", required=True, help="Path to the validated metadata CSV")
    parser.add_argument("--output_jsonl", required=True, help="Path to save the final JSONL dataset")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model version")
    args = parser.parse_args()

    generator = GeminiDatasetGenerator(api_key=args.api_key, model_id=args.model)
    generator.process_metadata_csv(args.input_csv, args.output_jsonl)

if __name__ == "__main__":
    main()
