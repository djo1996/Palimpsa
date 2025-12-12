import argparse
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu subset for Flame training.")
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        required=True, 
        help="Path to store the dataset (e.g., /Local/your_name/.cache)"
    )
    args = parser.parse_args()

    print(f"Checking/Downloading dataset to {args.cache_dir}...")
    os.makedirs(args.cache_dir, exist_ok=True)

    # Download the 100BT sample
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-100BT", 
        split="train",
        num_proc=16, 
        cache_dir=args.cache_dir
    )

    print(f"Success! Dataset is cached at {args.cache_dir} and ready for Flame.")

if __name__ == "__main__":
    main()