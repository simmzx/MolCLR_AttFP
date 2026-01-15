#!/bin/bash
#=============================================================================
# Monitoring and Utility Commands for MolCLR Experiment
#=============================================================================

PROJECT_DIR="/local-house/zhangxiang/MolCLR-master"
OUTPUT_DIR="${PROJECT_DIR}/ckpt_attentivefp"
LOG_DIR="${PROJECT_DIR}/logs_finetune"

echo "=============================================="
echo "MolCLR Experiment Monitoring Commands"
echo "=============================================="
echo ""

echo "1. Check GPU status:"
echo "   nvidia-smi"
echo "   watch -n 1 nvidia-smi"
echo ""

echo "2. Monitor pretraining progress:"
echo "   tail -f ${OUTPUT_DIR}/molclr_attfp_*/pretrain.log"
echo "   tail -f pretrain.out"
echo ""

echo "3. Monitor finetuning progress:"
echo "   tail -f ${LOG_DIR}/*.log"
echo "   tail -f finetune.out"
echo ""

echo "4. Check running processes:"
echo "   ps aux | grep python"
echo "   ps aux | grep molclr"
echo ""

echo "5. View AUROC results:"
echo "   cat ${OUTPUT_DIR}/*_auroc.txt"
echo ""

echo "6. Kill background job:"
echo "   jobs -l"
echo "   kill %1"
echo "   # Or find PID: ps aux | grep molclr"
echo "   # Then: kill <PID>"
echo ""

echo "7. Check disk usage:"
echo "   du -sh ${OUTPUT_DIR}/*"
echo ""

echo "8. View latest checkpoint:"
echo "   ls -lt ${OUTPUT_DIR}/*.pth | head -5"
echo ""

echo "=============================================="
echo "Quick Start Commands"
echo "=============================================="
echo ""

echo "# Step 1: Copy files to project directory"
echo "cp molclr_pretrain.py ${PROJECT_DIR}/"
echo "cp molclr_finetune.py ${PROJECT_DIR}/"
echo "cp run_pretrain.sh ${PROJECT_DIR}/"
echo "cp run_finetune.sh ${PROJECT_DIR}/"
echo ""

echo "# Step 2: Start pretraining in background"
echo "cd ${PROJECT_DIR}"
echo "chmod +x run_pretrain.sh"
echo "nohup ./run_pretrain.sh > pretrain.out 2>&1 &"
echo ""

echo "# Step 3: Monitor pretraining"
echo "tail -f pretrain.out"
echo ""

echo "# Step 4: After pretraining, start finetuning"
echo "chmod +x run_finetune.sh"
echo "nohup ./run_finetune.sh > finetune.out 2>&1 &"
echo ""

echo "# Step 5: Check final results"
echo "cat ${OUTPUT_DIR}/*_auroc.txt | grep -E 'TS2|TS3'"
echo ""
