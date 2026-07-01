"""
Generate TSV files compatible with the training script from your CSV files.

This creates TSV files with the required columns:
- vid: video/sample identifier (folder_name_sentence_id)
- split: train/valid/test
- raw-text: the text transcription
"""

import pandas as pd
import argparse
import os
from tqdm import tqdm


def read_text_file(text_file_path):
    """Read all lines from a text file."""
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def generate_tsv_for_split(csv_file, text_root, video_root, split_name, verify_videos=False):

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")

    required_columns = ["folder_name", "Sentence ID"]
    for c in required_columns:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    records = []
    split = split_name
    for idx, row in df.iterrows():
        folder = row["folder_name"]
        sent_id = int(row["Sentence ID"])     # sentence_id in CSV must be 1-based
        

        # Path to text file: <text-root>/<folder>/<folder>.txt
        text_file = os.path.join(text_root, folder, f"{folder}_sentences_clean.txt")

        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Missing text file: {text_file}")

        # Read all lines once
        with open(text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]

        # Convert to 0-based index
        zero_idx = sent_id - 1

        if zero_idx >= len(lines):
            raise IndexError(
                f"sentence_id {sent_id} out of range for file {text_file} "
                f"(file has {len(lines)} lines)"
            )

        sentence = lines[zero_idx]

        video_path = os.path.join(video_root, split_name, f'segment_{sent_id}_{folder}')

        records.append({
            "folder_name": folder,
            "sentence_id": sent_id,
            "split": split,
            "video_name": f'segment_{sent_id}_{folder}',
            "raw-text": sentence,
            "video_path": video_path
        })

    # Save output
    # out_df = pd.DataFrame(records)
    # out_df.to_csv(output_tsv, sep="\t", index=False)
    
    # Create DataFrame
    tsv_df = pd.DataFrame(records, columns=['folder_name', 'sentence_id', 'split', 'video_name', 'raw-text', 'video_path'])
    
    print(f"✓ Created {len(tsv_df)} entries for {split_name}")
    
    # if missing_text_files:
    #     print(f"⚠ {len(missing_text_files)} text files not found")
    
    # if verify_videos and missing_videos:
    #     print(f"⚠ {len(missing_videos)} videos not found")
    
    return tsv_df


def main():
    parser = argparse.ArgumentParser(description='Generate TSV files from CSV data')
    parser.add_argument('--text-root', required=True,
                        help='Root directory containing text files')
    parser.add_argument('--train-csv', required=True,
                        help='Path to train CSV file')
    parser.add_argument('--dev-csv', required=True,
                        help='Path to dev CSV file')
    parser.add_argument('--test-csv', required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output-tsv', required=True,
                        help='Output TSV file path (will contain all splits)')
    parser.add_argument('--verify-videos', action='store_true',
                        help='Verify that video files exist')
    parser.add_argument('--video-root', default=None,
                        help='Root directory for videos (needed if --verify-videos)')
    
    args = parser.parse_args()
    
    if args.verify_videos and not args.video_root:
        parser.error("--video-root is required when --verify-videos is set")
    
    print("="*60)
    print("GENERATING TSV FILES")
    print("="*60)
    print(f"Text root: {args.text_root}")
    print(f"Output TSV: {args.output_tsv}")
    if args.verify_videos:
        print(f"Video root: {args.video_root}")
        print("Video verification: ENABLED")
    print("="*60)
    
    # Process each split
    splits = [
        ('train', args.train_csv),
        ('dev', args.dev_csv),  # Note: using 'valid' to match the training script
        ('test', args.test_csv)
    ]
    
    output_dir = args.output_tsv
    for split_name, csv_file in splits:
        tsv_df = generate_tsv_for_split(
            csv_file=csv_file,
            text_root=args.text_root,
            video_root=args.video_root,
            split_name=split_name,
            verify_videos=args.verify_videos,
        )
        # Create text files and path
        
        with open(os.path.join(output_dir, f'dipolelo.{split_name}.en'), "w") as f1, open(os.path.join(output_dir, f'dipolelo.{split_name}.sign'), "w") as f2:
            for i, data in tsv_df.iterrows():
                if i == len(tsv_df) - 1:      # last item → no newline
                    f1.write(data['raw-text'])
                    f2.write(data['video_path'])
                else:                        # all others → include newline
                    f1.write(data['raw-text'] + "\n")
                    f2.write(data['video_path'] + "\n")


    
    
    # Save to TSV
    
    



if __name__ == '__main__':
    main()
