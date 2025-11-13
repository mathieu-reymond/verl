"""
Preprocess hiyouga/math12k dataset for VERL training.

This script converts the math12k dataset into the format required by VERL:
- Adds instruction wrapping with \\boxed{} format
- Adds data_source field for reward function routing
- Adds reward_model field
- Splits into train/test parquet files

The dataset contains 12,000 math problems with numerical answers.
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer


def preprocess_math12k(output_dir: str = None):
    """
    Preprocess the hiyouga/math12k dataset.
    
    Args:
        output_dir: Directory to save the processed parquet files.
                   Defaults to $HF_HOME/data/math12k or ./data/math12k
    """
    # Determine output directory
    if output_dir is None:
        hf_home = os.environ.get('HF_HOME', '.')
        output_dir = os.path.join(hf_home, 'data', 'math12k')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load the dataset
    print("Loading hiyouga/math12k dataset...")
    dataset = load_dataset("hiyouga/math12k")
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    # Instruction following - using \boxed{} format for consistency with math_verify and math-rl-16k
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    
    def process_example(example):
        """Process a single example."""
        question = example['problem'].strip() + "\n\n" + instruction_following
        
        return {
            'data_source': 'hiyouga/math12k',
            'prompt': [
                {
                    'role': 'user',
                    'content': question,
                }
            ],
            'ability': 'math',
            'reward_model': {
                'style': 'rule',
                'ground_truth': example['answer'],
            }
        }
    
    # Process train and test splits
    print("\nProcessing train split...")
    train_data = dataset['train'].map(
        process_example,
        remove_columns=dataset['train'].column_names,
        desc="Processing train examples"
    )
    
    print("Processing test split...")
    test_data = dataset['test'].map(
        process_example,
        remove_columns=dataset['test'].column_names,
        desc="Processing test examples"
    )
    
    # Save to parquet
    train_output = os.path.join(output_dir, 'train.parquet')
    test_output = os.path.join(output_dir, 'test.parquet')
    
    print(f"\nSaving train data to {train_output}")
    train_data.to_parquet(train_output)
    
    print(f"Saving test data to {test_output}")
    test_data.to_parquet(test_output)
    
    print("\n✓ Preprocessing complete!")
    print(f"  Train: {len(train_data)} examples → {train_output}")
    print(f"  Test: {len(test_data)} examples → {test_output}")
    
    # Show a sample
    print("\nSample processed example:")
    print(f"  data_source: {train_data[0]['data_source']}")
    print(f"  prompt: {train_data[0]['prompt'][0]['content'][:150]}...")
    print(f"  reward_model: {train_data[0]['reward_model']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess hiyouga/math12k dataset')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for parquet files (default: $HF_HOME/data/math12k)'
    )
    
    args = parser.parse_args()
    preprocess_math12k(output_dir=args.output_dir)
