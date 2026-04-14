"""
Main entry point for the Satellite Change Detection Pipeline.
Orchestrates the end-to-end flow from raw images to a validated VLM dataset.
"""

import argparse
from segmentation_core import ChangeDetectionPipeline
from best_of_n_retrieval import BestOfNVerifier
from dataset_construction import DatasetConstructor

def run_pipeline(args):
    # 1. Initialize stages
    cd_pipeline = ChangeDetectionPipeline()
    verifier = BestOfNVerifier(model_id=args.reward_model)
    formatter = DatasetConstructor(args.output_file)

    # 2. Process image pairs (Example loop)
    # In a real scenario, this would iterate over your LEVIR subset
    for img1, img2 in load_subset(args.input_dir):
        # Candidate Generation & CLIP Screening
        candidates = cd_pipeline.run_inference(img1, img2)
        
        # Best-of-N Verification
        # (Assuming you have a reference dataframe for RAG)
        validated_changes = verifier.select_best_of_n(
            candidates, img1, img2, examples_df=args.examples_path
        )
        
        # 3. Format & Save
        for change in validated_changes:
            formatter.create_instruction_pair(img1, img2, change=change)
            
    formatter.save_jsonl()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--reward_model", default="google/gemma-3-4b-it")
    parser.add_argument("--output_file", default="final_dataset.jsonl")
    args = parser.parse_args()
    run_pipeline(args)
