#!/bin/bash

# Configuration
TEXT_ROOT="../../../M2M-MT-files/Youtube_Download_2_Transcription_Whisper"
VIDEO_ROOT="M2M-MT-files/Youtube_Dipolelo_Video_Frames_Resized"

# CSV files
TRAIN_CSV="../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/train_sentences.csv"
DEV_CSV="../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/dev_sentences.csv"
TEST_CSV="../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/test_sentences.csv"

# Output TSV fileclear
OUTPUT_TSV="../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames_Paths"

echo "Generating TSV file for training..."
echo ""

python ./create_train_data_for_SignJoey.py \
    --text-root "${TEXT_ROOT}" \
    --train-csv "${TRAIN_CSV}" \
    --dev-csv "${DEV_CSV}" \
    --test-csv "${TEST_CSV}" \
    --output-tsv "${OUTPUT_TSV}" \
    --verify-videos \
    --video-root "${VIDEO_ROOT}"

echo ""
echo "Done! TSV file created at: ${OUTPUT_TSV}"
echo ""
echo "You can now use this TSV file with the training script:"
echo "  --label_path ${OUTPUT_TSV}"
