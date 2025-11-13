"""
Preprocess Open-Reasoner-Zero/orz_math_57k_collected dataset for VERL training.

This script converts the ORZ Math 57k dataset into the format required by VERL:
- Adds instruction wrapping with \\boxed{} format
- Adds data_source field for reward function routing
- Adds reward_model field
- Splits into train/val parquet files

The dataset contains 56,878 math problems with ground truth answers.
Dataset: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_57k_collected.json
"""

import os
import json
import requests
from datasets import Dataset
import argparse


def preprocess_orz_math_57k(output_dir: str = None, split_ratio: float = 0.95):
    """
    Preprocess the ORZ Math 57k dataset.
    
    Args:
        output_dir: Directory to save the processed parquet files.
                   Defaults to $HF_HOME/data/orz_math_57k or ./data/orz_math_57k
        split_ratio: Ratio for train/val split (default: 0.95)
    """
    # Determine output directory
    if output_dir is None:
        hf_home = os.environ.get('HF_HOME', '.')
        output_dir = os.path.join(hf_home, 'data', 'orz_math_57k')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Download the dataset
    print("Downloading ORZ Math 57k dataset...")
    url = "https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/main/data/orz_math_57k_collected.json"
    response = requests.get(url)
    data = json.loads(response.text)
    
    print(f"Total examples: {len(data)}")
    
    # Instruction following - using \boxed{} format for consistency with math_verify and math-rl-16k
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    
    def process_example(conversation):
        """Process a single conversation."""
        # Extract the problem from human message
        problem = conversation[0]['value']
        
        # Extract the ground truth from assistant message
        ground_truth = conversation[1]['ground_truth']['value']
        
        # Add instruction to the problem
        question = problem.strip() + "\n\n" + instruction_following
        
        return {
            'data_source': 'open-reasoner-zero/orz_math_57k',
            'prompt': [
                {
                    'role': 'user',
                    'content': question,
                }
            ],
            'ability': 'math',
            'reward_model': {
                'style': 'rule',
                'ground_truth': ground_truth,
            }
        }
    
    # Process all examples
    print("\nProcessing examples...")
    processed_data = [process_example(conv) for conv in data]
    
    # Split into train and validation
    split_idx = int(len(processed_data) * split_ratio)
    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Save to parquet
    train_output = os.path.join(output_dir, 'train.parquet')
    val_output = os.path.join(output_dir, 'test.parquet')
    
    print(f"\nSaving train data to {train_output}")
    train_dataset.to_parquet(train_output)
    
    print(f"Saving validation data to {val_output}")
    val_dataset.to_parquet(val_output)
    
    print("\n✓ Preprocessing complete!")
    print(f"  Train: {len(train_dataset)} examples → {train_output}")
    print(f"  Validation: {len(val_dataset)} examples → {val_output}")
    
    # Show a sample
    print("\nSample processed example:")
    print(f"  data_source: {train_dataset[0]['data_source']}")
    print(f"  prompt: {train_dataset[0]['prompt'][0]['content'][:150]}...")
    print(f"  reward_model: {train_dataset[0]['reward_model']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ORZ Math 57k dataset')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for parquet files (default: $HF_HOME/data/orz_math_57k)'
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.95,
        help='Ratio of train split. The rest will be used for validation (default: 0.95)'
    )
    
    args = parser.parse_args()
    preprocess_orz_math_57k(output_dir=args.output_dir, split_ratio=args.split_ratio)
