#!/bin/bash
#=============================================================================
# MolCLR Finetuning Script (Background Mode)
# 
# Usage:
#   chmod +x run_finetune.sh
#   nohup ./run_finetune.sh > finetune.out 2>&1 &
#   
# Monitor:
#   tail -f finetune.out
#   tail -f /local-house/zhangxiang/MolCLR-master/logs_finetune/*.log
#=============================================================================

# Configuration
PROJECT_DIR="/local-house/zhangxiang/MolCLR-master"
DATA_DIR="/local-house/zhangxiang/attrmasking_attentivefp/data"
OUTPUT_DIR="${PROJECT_DIR}/ckpt_attentivefp"

# Pretrained model path (update this after pretraining)
# Option 1: Specify the exact path
PRETRAINED_MODEL="${OUTPUT_DIR}/gnn_pretrained.pth"

# Option 2: Find the latest pretrained model automatically
# PRETRAINED_MODEL=$(find ${OUTPUT_DIR} -name "gnn_pretrained.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")

# Finetuning dataset
FINETUNE_DATA="${DATA_DIR}/train_dataset/finetune/finetune_800k.csv"

# GPU Selection
GPU_ID=3

# Experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="molclr_finetune_${TIMESTAMP}"

echo "=============================================="
echo "MolCLR + AttentiveFP Finetuning"
echo "=============================================="
echo "Start: $(date)"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "Finetune data: ${FINETUNE_DATA}"
echo "GPU: ${GPU_ID}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo ""

# Check if pretrained model exists
if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: Pretrained model not found: ${PRETRAINED_MODEL}"
    echo "Please run pretraining first or update the path."
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Run finetuning
cd ${PROJECT_DIR}

python molclr_finetune.py \
    --input_model_file ${PRETRAINED_MODEL} \
    --dataset ${FINETUNE_DATA} \
    --device 0 \
    --experiment_name ${EXPERIMENT_NAME}

echo ""
echo "=============================================="
echo "Finetuning Completed!"
echo "=============================================="
echo "End: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}_auroc.txt"
