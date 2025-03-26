import os
import shutil
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import random_split
import sys

def preprocess_and_save_hdf5(dataset_dir, output_dir, train_ratio=0.9, validate_ratio=0.05, test_ratio=0.05, low_res_size=(16, 16), high_res_size=(256, 256)):
    """
    Splits dataset into train/validate/test sets and saves them as separate HDF5 files with low/high resolution pairs.
    """
    assert train_ratio + validate_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    all_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    total_files = len(all_files)
    
    train_len = int(total_files * train_ratio)
    validate_len = int(total_files * validate_ratio)
    test_len = total_files - train_len - validate_len
    
    train_files, validate_files, test_files = random_split(all_files, [train_len, validate_len, test_len])
    
    for split_name, split_files in zip(["train", "validate", "test"], [train_files, validate_files, test_files]):
        output_h5_file = os.path.join(output_dir, f"{split_name}.h5")
        print(f"Processing {split_name} set...")
        
        with h5py.File(output_h5_file, "w") as h5f:
            low_res_group = h5f.create_group("low_res")
            high_res_group = h5f.create_group("high_res")
            
            for i, file_name in enumerate(tqdm(split_files, desc=f"{split_name.capitalize()} data")):
                file_path = os.path.join(dataset_dir, file_name)
                
                image = Image.open(file_path).convert("RGB")
                high_res = np.array(image.resize(high_res_size, Image.BICUBIC), dtype=np.uint8)  # High resolution
                low_res = np.array(image.resize(low_res_size, Image.BICUBIC), dtype=np.uint8)  # Low resolution
                
                low_res_group.create_dataset(str(i), data=low_res)
                high_res_group.create_dataset(str(i), data=high_res)
        
        print(f"{split_name.capitalize()} HDF5 dataset saved at {output_h5_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_test_split.py <dataset_dir> <output_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    preprocess_and_save_hdf5(dataset_dir, output_dir)