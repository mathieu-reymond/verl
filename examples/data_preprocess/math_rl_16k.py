# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the riddickz/math-rl-16k dataset to parquet format.
This dataset contains mathematical problems with verified solutions.

Note: This dataset uses math_verify for reward computation. To use it:
1. Install math_verify: pip install math-verify
2. Update verl/utils/reward_score/__init__.py to use math_verify for this data_source
   OR use a custom reward function that calls math_verify.compute_score()
"""

import argparse
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="(Deprecated) Use --local_save_dir instead")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for remote storage")
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math_rl_16k", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--split_ratio", 
        type=float, 
        default=0.95, 
        help="Ratio of train split. The rest will be used for validation (default: 0.95)"
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "riddickz/math-rl-16k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # The dataset only has a 'train' split, so we'll split it manually
    full_dataset = dataset["train"]

    # Instruction following - using \boxed{} format for consistency with math_verify
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")
            question = question_raw.strip() + "\n\n" + instruction_following

            solution = example.pop("solution")
            
            # Clean up the remaining fields we don't need for training
            example.pop("reward", None)
            example.pop("question_token_count", None)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "solution": solution,
                },
            }
            return data

        return process_fn

    # Split the dataset into train and validation
    split_dataset = full_dataset.train_test_split(
        test_size=(1 - args.split_ratio), 
        seed=42,  # Fixed seed for reproducibility
        shuffle=True
    )
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Total examples: {len(full_dataset)}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"\nDataset saved to: {local_save_dir}")
    print(f"  - train.parquet: {len(train_dataset)} examples")
    print(f"  - test.parquet: {len(val_dataset)} examples")
    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
        print(f"Dataset copied to HDFS: {hdfs_dir}")
