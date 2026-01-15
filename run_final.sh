#!/bin/bash
#=============================================================================
# MolCLR Pretraining - FINAL STABLE VERSION (GPU 6)
#=============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}

DATA_PATH="/local-house/zhangxiang/attrmasking_attentivefp/data/pretrain/smiles_1m.txt"
OUTPUT_DIR="${SCRIPT_DIR}/ckpt_molclr"

# GPU 6 is completely free
export CUDA_VISIBLE_DEVICES=6

EPOCHS=20
BATCH_SIZE=256
LR=0.0005
NUM_WORKERS=4

echo "=============================================="
echo "MolCLR Pretraining (FINAL STABLE VERSION)"
echo "=============================================="
echo "Start: $(date)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}"
echo "=============================================="

mkdir -p ${OUTPUT_DIR}

python molclr_pretrain_final.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --device 0 \
    --num_workers ${NUM_WORKERS}

echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
