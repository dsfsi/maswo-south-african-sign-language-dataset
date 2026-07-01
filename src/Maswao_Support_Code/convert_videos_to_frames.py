import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_rate=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        frame_rate: If specified, extract every Nth frame. If None, extract all frames.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame based on frame_rate parameter
        if frame_rate is None or frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def process_dataset(csv_path, video_base_dir, output_base_dir, dataset_name, frame_rate=None):
    """
    Process a dataset (train/dev/test) by extracting frames from videos.
    
    Args:
        csv_path: Path to CSV/Excel file containing video names
        video_base_dir: Base directory where videos are stored
        output_base_dir: Base directory for output folders
        dataset_name: Name of dataset (train/dev/test)
        frame_rate: If specified, extract every Nth frame
    """
    # Read the spreadsheet
    if csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path)
    elif csv_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(csv_path)
    else:
        raise ValueError("File must be CSV or Excel format")
    
    # Check if required columns exist
    if 'Sentence ID' not in df.columns or 'folder_name' not in df.columns:
        raise ValueError(f"CSV must contain 'sentence ID' and 'folder_name' columns. Found columns: {df.columns.tolist()}")
    
    # Create dataset output directory
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    print(f"\nProcessing {dataset_name} dataset...")
    print(f"Found {len(df)} videos")
    
    # Process each video
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
        sentence_id = row['Sentence ID']
        folder_name = row['folder_name']
        
        # Construct video filename: segment_{sentence_ID}_{folder_name}.mp4
        video_filename = f"segment_{sentence_id}_{folder_name}.mp4"
        video_path = os.path.join(video_base_dir, folder_name, "segment_gradients_sent_ratios_1", video_filename)
        
        # Create video-specific output folder
        video_folder_name = f"segment_{sentence_id}_{folder_name}"
        video_folder = os.path.join(dataset_output_dir, video_folder_name)
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_filename}")
            continue
        
        # Extract frames
        num_frames = extract_frames(video_path, video_folder, frame_rate)
        print(f"  Extracted {num_frames} frames from {video_filename}")

def main():
    """
    Main function to process all datasets.
    Configure your paths here.
    """
    # ========== CONFIGURATION ==========
    # Paths to your CSV/Excel files
    train_csv = "../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/train_sentences.csv"  # or "train.xlsx"
    dev_csv = "../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/dev_sentences.csv"
    test_csv = "../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/test_sentences.csv"
    
    # Directory where your video files are stored
    video_base_directory = "../../../M2M-MT-files/Youtube_Downloads_2/English"
    
    # Output directory where frames will be saved
    output_directory = "../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames"
    
    # Frame extraction rate (None = all frames, 30 = every 30th frame, etc.)
    frame_extraction_rate = None  # Change to 30, 60, etc. to sample frames
    # ===================================
    
    # Process each dataset
    datasets = [
        (train_csv, "train"),
        (dev_csv, "dev"),
        (test_csv, "test")
    ]
    
    for csv_path, dataset_name in datasets:
        if os.path.exists(csv_path):
            try:
                process_dataset(
                    csv_path, 
                    video_base_directory, 
                    output_directory, 
                    dataset_name,
                    frame_extraction_rate
                )
            except Exception as e:
                print(f"Error processing {dataset_name}: {str(e)}")
        else:
            print(f"Warning: {csv_path} not found, skipping {dataset_name} dataset")
    
    print("\nProcessing complete!")
    print(f"Frames saved to: {output_directory}")

if __name__ == "__main__":
    main()