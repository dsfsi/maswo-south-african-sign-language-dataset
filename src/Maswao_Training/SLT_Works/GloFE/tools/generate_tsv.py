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


def generate_tsv_for_split(csv_file, text_root, split_name, verify_videos=False, video_root=None):

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

        records.append({
            "folder_name": folder,
            "sentence_id": sent_id,
            "split": split,
            "video_name": f'segment_{sent_id}_{folder}',
            "raw-text": sentence
        })

    # Save output
    # out_df = pd.DataFrame(records)
    # out_df.to_csv(output_tsv, sep="\t", index=False)
    
    # Create DataFrame
    tsv_df = pd.DataFrame(records, columns=['folder_name', 'sentence_id', 'split', 'video_name', 'raw-text'])
    
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
    
    all_data = []
    all_missing_text = []
    all_missing_videos = []
    
    for split_name, csv_file in splits:
        tsv_df = generate_tsv_for_split(
            csv_file=csv_file,
            text_root=args.text_root,
            split_name=split_name,
            verify_videos=args.verify_videos,
            video_root=args.video_root
        )
        all_data.append(tsv_df)
        # all_missing_text.extend(missing_text)
        # all_missing_videos.extend(missing_videos)
    
    # Combine all splits into one TSV
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to TSV
    output_dir = os.path.dirname(args.output_tsv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    combined_df.to_csv(args.output_tsv, sep='\t', index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total entries: {len(combined_df)}")
    print(f"  Train: {len(combined_df[combined_df['split'] == 'train'])}")
    print(f"  Valid: {len(combined_df[combined_df['split'] == 'dev'])}")
    print(f"  Test: {len(combined_df[combined_df['split'] == 'test'])}")
    print(f"\nOutput saved to: {args.output_tsv}")
    
    # if all_missing_text:
    #     print(f"\n⚠ {len(all_missing_text)} text files not found")
    #     error_log = args.output_tsv.replace('.tsv', '_missing_text.txt')
    #     with open(error_log, 'w') as f:
    #         f.write('\n'.join(all_missing_text))
    #     print(f"  Missing text log: {error_log}")
    
    # if all_missing_videos:
    #     print(f"\n⚠ {len(all_missing_videos)} videos not found")
    #     error_log = args.output_tsv.replace('.tsv', '_missing_videos.txt')
    #     with open(error_log, 'w') as f:
    #         f.write('\n'.join(all_missing_videos))
    #     print(f"  Missing video log: {error_log}")
    
    print("="*60)
    print("\n✓ TSV generation complete!")
    print(f"\nTSV file format preview:")
    print(combined_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
