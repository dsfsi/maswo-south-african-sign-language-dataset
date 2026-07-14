
#!/bin/bash

# GPU Configuration
GPUS=4

# Paths Configuration
WORK_DIR_PREFIX="../../../M2M-MT-files"
WORK_DIR="m2m_sign_translation_batch_64"

# Data Paths - MODIFY THESE FOR YOUR DATASET
# Note: The training script uses a single TSV file with split column
# Generate this TSV using generate_tsv.sh first
FEAT_PATH="../../../M2M-MT-files/Youtube_MMPose"  # Contains train/dev/test subdirectories
LABEL_PATH="../../../M2M-MT-files/Youtube_CSV_files/dipolelo_dataset_labels.tsv"  # Single TSV with all splits

# Tokenizer and vocabulary paths (you'll need to train/obtain these)
TOKENIZER_PATH="../../notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased"  # TODO: Train your tokenizer
WORD_EMBED_PATH="../../notebooks/openasl-v1.0/uncased_filtred_glove_VN_embed.pkl"  # TODO: Prepare embeddings (optional)

# Training Hyperparameters
BATCH_SIZE=64
LABEL_SMOOTHING=0.2
EPOCHS=400
SAVE_EVERY=5
CLIP_LENGTH=512
VOCAB_SIZE=25000

# Model Configuration
POSE_BACKBONE="PartedPoseBackbone"
LEARNING_RATE=3e-4
DROPOUT_ENC=0.1
DROPOUT_DEC=0.1

# Inter-contrastive Learning
INTER_CL_MARGIN=0.0
INTER_CL_ALPHA=0.0
INTER_CL_VOCAB=5523

# EOS Token
EOS_TOKEN="</s>"

echo "========================================="
echo "Training Configuration"
echo "========================================="
echo "GPUs: $GPUS"
echo "Work Directory: $WORK_DIR_PREFIX/$WORK_DIR"
echo ""
echo "Data Paths:"
echo "  Train Features: $FEAT_PATH"
# echo "  Dev Features:   $DEV_FEAT_PATH"
# echo "  Test Features:  $TEST_FEAT_PATH"
# echo "  Train Labels:   $TRAIN_LABEL_PATH"
# echo "  Dev Labels:     $DEV_LABEL_PATH"
echo "  Test Labels:    $LABEL_PATH"
echo ""
echo "Training Params:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "========================================="
echo ""

# Run training with distributed data parallel
torchrun --nnodes=1 --nproc_per_node=$GPUS ../../train_dipolelo_pose_DDP_inter_VN.py \
    --ngpus $GPUS \
    --work_dir_prefix "$WORK_DIR_PREFIX" \
    --work_dir "$WORK_DIR" \
    --bs $BATCH_SIZE \
    --ls $LABEL_SMOOTHING \
    --epochs $EPOCHS \
    --save_every $SAVE_EVERY \
    --clip_length $CLIP_LENGTH \
    --vocab_size $VOCAB_SIZE \
    --feat_path "$FEAT_PATH" \
    --label_path "$LABEL_PATH" \
    --eos_token "$EOS_TOKEN" \
    --tokenizer "$TOKENIZER_PATH" \
    --pose_backbone "$POSE_BACKBONE" \
    --pe_enc \
    --mask_enc \
    --lr $LEARNING_RATE \
    --dropout_dec $DROPOUT_DEC \
    --dropout_enc $DROPOUT_ENC \
    --inter_cl_we_path "$WORD_EMBED_PATH"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
