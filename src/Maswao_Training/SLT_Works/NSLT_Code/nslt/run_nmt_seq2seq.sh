#!/bin/bash

# ==== User-defined variables ====
OUT_DIR="../../../../../M2M-MT-files/NSLT_Dipolelo_Benchmark"
GPU_ID="0"

# ==== Run NMT training ====
python -m nmt \
    --src=sign \
    --tgt=en \
    --train_prefix=../../../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames_Paths/dipolelo.train \
    --dev_prefix=../../../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames_Paths/dipolelo.dev \
    --test_prefix=../../../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames_Paths/dipolelo.test \
    --out_dir="${OUT_DIR}" \
    --vocab_prefix=../../../../../M2M-MT-files/Youtube_Dipolelo_Video_Frames_Paths/dipolelo.VOCAB_original \
    --source_reverse=True \
    --num_units=1000 \
    --num_layers=4 \
    --num_train_steps=150000 \
    --residual=True \
    --attention=luong \
    --base_gpu="${GPU_ID}" \
    --unit_type=gru \
    --num_gpus=2
