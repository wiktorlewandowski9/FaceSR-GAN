import os
from PIL import Image
import argparse
from tqdm import tqdm
import random

def resize_image(image_path, size):
    image = Image.open(image_path).convert("RGB")
    return image.resize(size, Image.BICUBIC)

def split_and_prepare_dataset(dataset_dir, output_dir, train_ratio, validate_ratio, test_ratio, high_res_size):
    assert train_ratio + validate_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    assert high_res_size % 4 == 0, "High-res size must be divisible by 4."

    # List all image files
    all_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    if not all_files:
        raise ValueError("Input directory is empty or contains no valid files.")
    
    # Shuffle files for randomness
    random.shuffle(all_files)
    total = len(all_files)

    # Split dataset
    train_len = int(total * train_ratio)
    validate_len = int(total * validate_ratio)
    test_len = total - train_len - validate_len

    train_files = all_files[:train_len]
    validate_files = all_files[train_len:train_len + validate_len]
    test_files = all_files[train_len + validate_len:]

    splits = {
        "train": train_files,
        "validate": validate_files,
        "test": test_files
    }

    # Create output structure
    subsets = ['train', 'validate', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, "high_res"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "low_res"), exist_ok=True)

    for split_name, file_list in splits.items():
        print(f"Processing {split_name} files...")
        for file in tqdm(file_list, desc=f"{split_name}"):
            src_path = os.path.join(dataset_dir, file)

            # Load image
            img = Image.open(src_path).convert("RGB")
            
            # Create and save high-res version
            high_img = img.resize((high_res_size, high_res_size), Image.BICUBIC)
            high_res_path = os.path.join(output_dir, split_name, "high_res", file)
            high_img.save(high_res_path)

            # Create and save low-res version
            low_img_size = (high_res_size // 4, high_res_size // 4)
            low_img = img.resize(low_img_size, Image.BICUBIC)
            low_res_path = os.path.join(output_dir, split_name, "low_res", file)
            low_img.save(low_res_path)

    print("✅ Dataset preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and generate paired low/high-res dataset.")
    parser.add_argument("--source", required=True, type=str, help="Path to source images.")
    parser.add_argument("--output", required=True, type=str, help="Path to output split dataset.")
    parser.add_argument("--train_size", type=float, default=0.9, help="Proportion of training data.")
    parser.add_argument("--validate_size", type=float, default=0.05, help="Proportion of validation data.")
    parser.add_argument("--test_size", type=float, default=0.05, help="Proportion of test data.")
    parser.add_argument("--high_res_size", type=int, default=256, help="Size of high-res images (must be divisible by 4).")

    args = parser.parse_args()

    split_and_prepare_dataset(
        dataset_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_size,
        validate_ratio=args.validate_size,
        test_ratio=args.test_size,
        high_res_size=args.high_res_size
    )