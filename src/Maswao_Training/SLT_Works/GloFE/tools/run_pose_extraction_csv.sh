#!/bin/bash
# Configuration
DET_CONFIG="~/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
DET_CHECKPOINT="~/mmpose/mmpose/models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
POSE_CONFIG="~/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py"
POSE_CHECKPOINT="~/mmpose/mmpose/models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

VIDEO_ROOT="../../M2M-MT-files/Youtube_Downloads_2/English"
OUTPUT_ROOT="../../M2M-MT-files/Youtube_MMPose"

# CSV files
TRAIN_CSV="../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/train_sentences.csv"
DEV_CSV="../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/dev_sentences.csv"
TEST_CSV="../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports/test_sentences.csv"

# Column names (adjust if different in your CSVs)
SEGMENT_COL="folder_name"
VIDEO_COL="Sentence ID"

# Parallel processing settings
NUM_SPLITS=4  # Number of parallel jobs per dataset
DEVICE="cuda:1"  # Change to cuda:1, cuda:2, etc. for multiple GPUs

# Optional parameters
BBOX_THR=0.3
KPT_THR=0.3
USE_MULTI_FRAMES=""  # Set to "--use-multi-frames" to enable

# Function to process a single CSV file with parallel splits
process_csv() {
    local CSV_FILE=$1
    local SPLIT_NAME=$2
    local OUTPUT_DIR="${OUTPUT_ROOT}/${SPLIT_NAME}"
    
    echo "========================================="
    echo "Processing: ${SPLIT_NAME}"
    echo "CSV: ${CSV_FILE}"
    echo "Output: ${OUTPUT_DIR}"
    echo "========================================="
    
    mkdir -p "${OUTPUT_DIR}"
    
    # Launch parallel jobs
    for i in $(seq 0 $((NUM_SPLITS - 1))); do
        echo "Launching split ${i}/${NUM_SPLITS} for ${SPLIT_NAME}..."
        
        python mmpose_pose_extraction_csv.py \
            "${DET_CONFIG}" \
            "${DET_CHECKPOINT}" \
            "${POSE_CONFIG}" \
            "${POSE_CHECKPOINT}" \
            --csv-file "${CSV_FILE}" \
            --video-root "${VIDEO_ROOT}" \
            --output-root "${OUTPUT_DIR}" \
            --segment-col "${SEGMENT_COL}" \
            --video-col "${VIDEO_COL}" \
            --device "${DEVICE}" \
            --bbox-thr ${BBOX_THR} \
            --kpt-thr ${KPT_THR} \
            --sid ${i} \
            --splits ${NUM_SPLITS} \
            ${USE_MULTI_FRAMES} \
            > "log_${SPLIT_NAME}_split${i}.txt" 2>&1 &
    done
    
    # Wait for all splits to complete
    echo "Waiting for all ${SPLIT_NAME} splits to complete..."
    wait
    echo "${SPLIT_NAME} processing completed!"
    echo ""
}

# Main execution
echo "Starting pose extraction for all datasets..."
echo ""

# Process each dataset sequentially (but with parallel splits within each)
process_csv "${TRAIN_CSV}" "train"
process_csv "${DEV_CSV}" "dev"
process_csv "${TEST_CSV}" "test"

echo "========================================="
echo "All datasets processed!"
echo "========================================="
echo ""
echo "Output locations:"
echo "  Train: ${OUTPUT_ROOT}/train"
echo "  Dev: ${OUTPUT_ROOT}/dev"
echo "  Test: ${OUTPUT_ROOT}/test"
echo ""
echo "Check log files for any errors:"
echo "  log_train_split*.txt"
echo "  log_dev_split*.txt"
echo "  log_test_split*.txt"
