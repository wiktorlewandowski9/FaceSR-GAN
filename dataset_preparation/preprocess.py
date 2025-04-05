import os
import shutil
from torch.utils.data import random_split
from tqdm import tqdm
import argparse

def split_dataset(dataset_dir, output_dir, train_ratio, validate_ratio, test_ratio):
    # Ensure the ratios sum to 1
    assert train_ratio + validate_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Check if input directory is empty
    if not os.listdir(dataset_dir):
        raise ValueError("Input directory is empty. Please provide a valid dataset.")

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    validate_dir = os.path.join(output_dir, 'validate')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validate_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all files in the dataset directory
    all_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

    # Calculate split lengths
    total_files = len(all_files)
    train_len = int(total_files * train_ratio)
    validate_len = int(total_files * validate_ratio)
    test_len = total_files - train_len - validate_len

    # Split files using torch
    train_files, validate_files, test_files = random_split(all_files, [train_len, validate_len, test_len])

    # Move files to respective directories with progress bars
    print("Moving training files...")
    for file in tqdm(train_files, desc="Train files"):
        shutil.move(os.path.join(dataset_dir, file), os.path.join(train_dir, file))
    
    print("Moving validation files...")
    for file in tqdm(validate_files, desc="Validation files"):
        shutil.move(os.path.join(dataset_dir, file), os.path.join(validate_dir, file))
    
    print("Moving test files...")
    for file in tqdm(test_files, desc="Test files"):
        shutil.move(os.path.join(dataset_dir, file), os.path.join(test_dir, file))

    print(f"Dataset split completed: {train_len} train, {validate_len} validate, {test_len} test files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to split a dataset into training, validation, and test sets."
    )
    parser.add_argument("--source", required=True, type=str, help="Path to the source dataset directory.")
    parser.add_argument("--output", required=True, type=str, help="Path to the output directory for the split dataset.")
    parser.add_argument("--train_size", required=False, type=float, help="Precent of training samples.", default=0.9)
    parser.add_argument("--validate_size", required=False, type=float, help="Percent of validation samples." , default=0.05)
    parser.add_argument("--test_size", required=False, type=float, help="Percent of test samples.", default=0.05)
    args = parser.parse_args()

    split_dataset(args.source, args.output, args.train_size, args.validate_size, args.test_size)