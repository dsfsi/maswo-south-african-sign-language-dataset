import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

def load_and_resize_image(image_path, target_size=(227, 227)):
    """Load an image and resize it to target size."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def compute_average_images_efficient(csv_path, root_folder, output_dir, 
                                     percentages=[0.25, 0.50, 0.75, 1.00], 
                                     target_size=(227, 227), random_seed=42):
    """
    Efficiently compute average images at different percentages in a single pass.
    
    Args:
        csv_path: Path to CSV file with folder_name and Sentence ID columns
        root_folder: Root directory containing train folder
        output_dir: Directory to save output files
        percentages: List of percentages to save at (e.g., [0.25, 0.5, 0.75, 1.0])
        target_size: Target image size (height, width)
        random_seed: Random seed for reproducibility
    """
    # Read and shuffle CSV
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    print(f"Total subfolders: {len(df)}")
    
    # Calculate milestone indices
    milestones = {}
    for pct in sorted(percentages):
        milestone_idx = int(len(df) * pct) - 1
        milestones[milestone_idx] = pct
    
    print(f"Milestones: {milestones}")
    
    # Initialize accumulator
    sum_image = np.zeros((*target_size, 3), dtype=np.float64)
    num_images = 0
    
    # Process each subfolder
    train_path = Path(root_folder)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing all data"):
        sentence_id = row['Sentence ID']
        folder_name = row['folder_name']
        
        # Construct subfolder path
        subfolder = train_path / f'segment_{sentence_id}_{folder_name}'
        
        if not subfolder.exists():
            print(f"Warning: Subfolder not found: {subfolder}")
            continue
        
        # Process all images in subfolder
        for img_file in subfolder.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_array = load_and_resize_image(img_file, target_size)
                if img_array is not None:
                    sum_image += img_array
                    num_images += 1
        
        # Check if we've reached a milestone
        if idx in milestones:
            pct = milestones[idx]
            if num_images > 0:
                # Compute average
                average_image = (sum_image / num_images).astype(np.uint8)
                
                # Save as .npy
                npy_filename = f'average_image_{int(pct*100)}pct.npy'
                npy_path = os.path.join(output_dir, npy_filename)
                np.save(npy_path, average_image)
                
                print(f"\n✓ Saved {int(pct*100)}% milestone: {npy_path}")
                print(f"  Shape: {average_image.shape}, Images processed: {num_images}")
            else:
                print(f"\n✗ No images found at {int(pct*100)}% milestone")
    
    return num_images

def main():
    # Configuration
    csv_path = '../../../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/train_sentences.csv'
    root_folder = '../../../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames/train'
    output_dir = '../../../../../M2M-MT-files/Avarage_Images_Dipolelo'
    random_seed = 42
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sampling percentages for ablation
    percentages = [0.25, 0.50, 0.75, 1.00]
    
    print("Computing average images for ablation study (efficient single-pass)...")
    print("=" * 70)
    
    total_images = compute_average_images_efficient(
        csv_path=csv_path,
        root_folder=root_folder,
        output_dir=output_dir,
        percentages=percentages,
        target_size=(227, 227),
        random_seed=random_seed
    )
    
    print("\n" + "=" * 70)
    print(f"All average images computed successfully!")
    print(f"Total images processed: {total_images}")

if __name__ == "__main__":
    main()